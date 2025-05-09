import torch
import visibility
import cv2
import math
import numpy as np
import time
from easydict import EasyDict as edict
from torch import Tensor
from .utils import get_logger, CameraIntrinsicParameters, Evaluation, register_evaluation, deproject, transform_distance, get_transform_from_rotation_translation, inverse_rotation_translation, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, rotation_vector_to_rotation_matrix, TransformDistanceType

MAX_FLOW = 400
EPSILON = 1e-9
FLOW_LOSS_THRESHOLD = 1e3
gamma = 0.8
A = 0.7


@register_evaluation
class SequenceLossFunction(Evaluation):
    def __init__(self, cfg: edict):
        super().__init__(cfg)

    def epe_fn(self, flow, info, flow_gt):
        epe_losses = []

        if not self._cfg.model.use_var:
            var_max = var_min = 0
        else:
            var_max = self._cfg.model.var_max
            var_min = self._cfg.model.var_min

        for i in range(len(info)):
            raw_b = info[i][:, 2:]
            log_b = torch.zeros_like(raw_b)
            weight = info[i][:, :2]
            log_b[:, 0] = torch.clamp(
                raw_b[:, 0], min=0, max=var_max
            )  # Large b Component
            log_b[:, 1] = torch.clamp(
                raw_b[:, 1], min=var_min, max=0
            )  # Small b Component
            term2 = ((flow_gt - flow[i]).abs().unsqueeze(2)) * (
                torch.exp(-log_b).unsqueeze(1)
            )  # term2: [N, 2, m, H, W]
            term1 = weight - math.log(2) - log_b  # term1: [N, m, H, W]
            epe_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(
                term1.unsqueeze(1) - term2, dim=2
            )
            epe_losses.append(epe_loss)
        return epe_losses

    def mask_fn(self, loss, mask):
        mask = (
            (~torch.isnan(loss.detach()))
            & (~torch.isinf(loss.detach()))
            & mask[:, None]
        )
        return (mask * loss).sum() / (mask.sum() + EPSILON)

    def evaluation_fn(self, data_dict, output_dict):
        """Loss function defined over sequence of flow predictions"""
        flow_loss = 0.0
        flow_gt = data_dict["flow_images_gt"]

        # Compute mask for valid flow
        mask_main = (torch.sum(flow_gt**2, dim=1).sqrt() < MAX_FLOW) & (
            flow_gt[:, 0, :, :] != 0
        ) | (flow_gt[:, 1, :, :] != 0)
        mask_feature = data_dict["depth_images_fine"][:, 0, :, :] != 0

        # Compute loss for each flow prediction
        epe_losses_main_pixel = self.epe_fn(
            output_dict["flow_main_pixel"], output_dict["info_main_pixel"], flow_gt
        )
        epe_losses_feature = self.epe_fn(
            output_dict["flow_feature"], output_dict["info_feature"], flow_gt
        )
        n_predictions = len(output_dict["flow_main_pixel"])
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_loss += i_weight * (
                self.mask_fn(epe_losses_main_pixel[i], mask_main).mean() * A
                + self.mask_fn(epe_losses_feature[i], mask_feature).mean() * (1 - A)
            )

        epe = torch.sum((output_dict["final"] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[mask_main.view(-1)]

        if flow_loss > FLOW_LOSS_THRESHOLD or torch.isnan(flow_loss):
            flow_loss = torch.tensor(0.0, requires_grad=True)

        return {
            "loss": flow_loss,
            "epe": epe.mean().item(),
            "1px": (epe < 1).float().mean().item(),
            "3px": (epe < 3).float().mean().item(),
            "5px": (epe < 5).float().mean().item(),
        }


@register_evaluation
class SequenceEvalFunction(Evaluation):
    def __init__(self, cfg: edict):
        super().__init__(cfg)
        self.val_metric = "val_epe"

    def evaluation_fn(self, data_dict, output_dict):
        flow_up = output_dict["final"]
        flow_gt = data_dict["flow_images_gt"]

        out_list, epe_list = [], []
        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)

        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
        val = valid_gt.view(-1) >= 0.5
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)

        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)
        return {"val_epe": epe, "val_f1": f1}


@register_evaluation
class FlowEvalFunction(Evaluation):
    def __init__(self, cfg: edict):
        super().__init__(cfg)

    def flow_image2transform_with_depth_image(
        self,
        flow_image: Tensor,
        depth_image: Tensor,
        camera_params: CameraIntrinsicParameters,
    ):
        device = flow_image.device
        # create output tensor and pred_depth_img tensor
        output = torch.zeros(flow_image.shape).to(device)
        pred_depth_img = torch.zeros(depth_image.shape).to(device)
        pred_depth_img += 1000.0
        # warp the depth image
        output: Tensor = visibility.image_warp_index(
            depth_image.to(device),
            flow_image.int(),
            pred_depth_img,
            output,
            depth_image.shape[3],
            depth_image.shape[2],
        )
        pred_depth_img[pred_depth_img == 1000.0] = 0.0
        pc_project_uv = output.cpu().permute(0, 2, 3, 1).numpy()
        depth_img_ori = depth_image.cpu().numpy() * 100.0

        # generate mask
        mask_depth_1 = pc_project_uv[0, :, :, 0] != 0
        mask_depth_2 = pc_project_uv[0, :, :, 1] != 0
        mask_depth = mask_depth_1 + mask_depth_2
        depth_img = depth_img_ori[0, 0, :, :] * mask_depth

        if (
            self._cfg.dataset.name == "DatasetKittiOdometry_I2P"
            or self._cfg.dataset.name == "Dataset_Sample"
        ):  # adjust the principal point 1241*376 W * H. 960*320
            h, w = 28, 140
        elif (
            self._cfg.dataset.name == "DatasetArgoverse_I2P"
        ):  # adjust the principal point 960*360 W * H. 960*320
            h, w = 20, 0
        elif (
            self._cfg.dataset.name == "DatasetNuscenes_I2P"
        ):  # adjust the principal point 960*360 W * H. 960*320
            h, w = 20, 0
        elif (
            self._cfg.dataset.name == "DatasetWaymo_I2P"
        ):  # adjust the principal point 960*384 W * H. 960*320
            h, w = 32, 0
        camera_params.principal_point_x = camera_params.principal_point_x - w
        camera_params.principal_point_y = camera_params.principal_point_y - h
        camera_params_matrix = camera_params.to_matrix().numpy()

        # deproject the depth image
        pts3d, pts2d, _ = deproject(depth_img, pc_project_uv[0, :, :, :], camera_params)
        start_time = time.time()
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(
            pts3d, pts2d, camera_params_matrix, None
        )
        get_logger().debug(f"solvePnPRansac time: {time.time() - start_time:.3f}s")
        # convert the rotation vector to euler angles
        rotation_matrix = rotation_vector_to_rotation_matrix(rvecs)
        translation_vector = torch.tensor(tvecs).float()
        rotation_predicted, translation_predicted = inverse_rotation_translation(
            rotation_matrix, translation_vector
        )
        rotation_predicted_quaternion = rotation_matrix_to_quaternion(
            rotation_predicted
        )
        if self._cfg.dataset.name in [
            "DatasetKittiOdometry_I2P",
            "DatasetWaymo_I2P",
            "Dataset_Sample",
        ]:
            rotation_predicted_quaternion = rotation_predicted_quaternion[
                0, [2, 0, 1, 3]
            ]
        rotation_predicted = quaternion_to_rotation_matrix(
            rotation_predicted_quaternion
        )
        if self._cfg.dataset.name in [
            "DatasetKittiOdometry_I2P",
            "DatasetWaymo_I2P",
            "Dataset_Sample",
        ]:
            translation_predicted = translation_predicted[:, [2, 0, 1]]

        transform_predicted = get_transform_from_rotation_translation(
            rotation_predicted, translation_predicted
        )
        return transform_predicted.to(device)

    def evaluation_fn(self, data_dict, output_dict):
        flow_image_predicted = output_dict["final"]
        depth_image = data_dict["depth_images_input"]
        camera_params = data_dict["camera_intrinsic_parameters"][0]
        translation_error = data_dict["tr_error"]
        rotation_error = data_dict["rot_error"]
        transform_predicted = self.flow_image2transform_with_depth_image(
            flow_image_predicted, depth_image, camera_params.clone()
        )
        rotation_distance, translation_distance = transform_distance(
            transform_predicted,
            get_transform_from_rotation_translation(rotation_error, translation_error),
            flag=TransformDistanceType.I2D_LOC,
        )
        return {
            "predict": transform_predicted,
            "Test_Trans_Error": translation_distance,
            "Test_Rotation_Error": rotation_distance,
        }
