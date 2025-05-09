import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from .layers import (
    conv3x3,
    coords_grid,
    CorrBlock,
    InputPadder,
    BasicUpdateBlock,
    ResNetFPN,
)
from .utils import register_model, EngineMode


@register_model
class I2D_LocX(nn.Module):
    def __init__(self, cfg: edict):
        super(I2D_LocX, self).__init__()
        self.cfg = cfg
        self.output_dim = cfg.model.dim * 2

        self.cfg.model.corr_levels = 4
        self.cfg.model.corr_radius = cfg.model.radius
        self.cfg.model.corr_channel = (
            cfg.model.corr_levels * (cfg.model.radius * 2 + 1) ** 2
        )

        self.cnet = ResNetFPN(
            cfg,
            input_dim=4,
            output_dim=self.cfg.model.dim * 2,
            norm_layer=nn.BatchNorm2d,
            init_weight=True,
        )
        self.init_conv = conv3x3(2 * cfg.model.dim, 2 * cfg.model.dim)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(cfg.model.dim, cfg.model.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.model.dim * 2, 64 * 9, 1, padding=0),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(cfg.model.dim, 2 * cfg.model.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * cfg.model.dim, 6, 3, padding=1),
        )  # flow(2) + weight(2) + log_b(2)

        if cfg.model.iters > 0:
            self.fnet = ResNetFPN(
                cfg,
                input_dim=3,
                output_dim=self.output_dim,
                norm_layer=nn.InstanceNorm2d,
                init_weight=True,
            )
            self.fnet_lidar = ResNetFPN(
                cfg,
                input_dim=1,
                output_dim=self.output_dim,
                norm_layer=nn.InstanceNorm2d,
                init_weight=True,
            )
            self.update_block = BasicUpdateBlock(
                cfg, hdim=cfg.model.dim, cdim=cfg.model.dim
            )

    def upsample_data(self, flow, info, mask):
        """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""

        def upsample(data, mask, channels):
            up_data = F.unfold(data, [3, 3], padding=1).view(N, channels, 9, 1, 1, H, W)
            up_data = torch.sum(mask * up_data, dim=2).permute(0, 1, 4, 2, 5, 3)
            return up_data.reshape(N, channels, 8 * H, 8 * W)

        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        return upsample(8 * flow, mask, 2), upsample(info, mask, C)

    def forward_branch(self, depth_image, vision_image, engine_mode=EngineMode.TRAIN):
        """Estimate optical flow between pair of frames"""

        N, _, H, W = depth_image.shape
        iters = self.cfg.model.iters
        depth_image = 2 * depth_image - 1.0
        depth_image = depth_image.contiguous()
        vision_image = vision_image.contiguous()

        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(depth_image.shape)
        depth_image, vision_image = padder.pad(depth_image, vision_image)
        dilation = torch.ones(N, 1, H // 8, W // 8, device=depth_image.device)

        # run the context network
        cnet = self.cnet(torch.cat([depth_image, vision_image], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(
            cnet, [self.cfg.model.dim, self.cfg.model.dim], dim=1
        )

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.cfg.model.iters > 0:
            fmap_depth_8x = self.fnet_lidar(depth_image)
            fmap_vision_8x = self.fnet(vision_image)
            corr_fn = CorrBlock(fmap_depth_8x, fmap_vision_8x, self.cfg)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (
                coords_grid(N, H, W, device=depth_image.device) + flow_8x
            ).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(flow_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        output_dict = {
            "final": flow_predictions[-1],
            "conf": info_predictions[-1][:, :2],
            "info": info_predictions[-1][:, 2:],
        }
        if engine_mode == EngineMode.TRAIN:
            output_dict.update(
                {
                    "flow": flow_predictions,
                    "info": info_predictions,
                }
            )
        return output_dict

    def forward(self, data_dict, engine_mode=EngineMode.TRAIN):
        """Estimate optical flow between pair of frames"""
        depth_image = data_dict["depth_images_input"]
        vision_image = data_dict["vision_images_input"]

        output_dict = {}
        output_dict_main = self.forward_branch(depth_image, vision_image, engine_mode)
        output_dict["final"] = output_dict_main["final"]
        output_dict["conf"] = output_dict_main["conf"]
        output_dict["info"] = output_dict_main["info"]

        if engine_mode == EngineMode.TRAIN:
            # output dict -> main branch output dict
            output_dict["flow_main_pixel"] = output_dict_main["flow"]
            output_dict["info_main_pixel"] = output_dict_main["info"]

            depth_image_fine = data_dict["depth_images_fine"]
            output_dict_feature = self.forward_branch(
                depth_image_fine, vision_image, engine_mode
            )
            # output dict -> zero-flow branch output dict
            output_dict["flow_feature"] = output_dict_feature["flow"]
            output_dict["info_feature"] = output_dict_feature["info"]

        return output_dict
