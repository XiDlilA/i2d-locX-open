from enum import Enum
import torch
from torch import Tensor
import visibility
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import random
from torch import initial_seed
import torch.nn as nn
from numpy import random as np_random
from scipy.spatial.transform import Rotation as R
from PIL import Image
import h5py
from numpy import ndarray
import pandas as pd
import torchvision.transforms.functional as TTF
from torchvision import transforms
import abc
from tqdm import tqdm
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel


class CameraIntrinsicParameters(Tensor):
    def __new__(cls, focal_length_x, focal_length_y, principal_point_x, principal_point_y):
        data = torch.tensor([focal_length_x, focal_length_y, principal_point_x, principal_point_y], dtype=torch.float32)
        return Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, focal_length_x, focal_length_y, principal_point_x, principal_point_y):
        pass

    @property
    def focal_length_x(self):
        return self[0]

    @focal_length_x.setter
    def focal_length_x(self, value):
        self[0] = value

    @property
    def focal_length_y(self):
        return self[1]

    @focal_length_y.setter
    def focal_length_y(self, value):
        self[1] = value

    @property
    def principal_point_x(self):
        return self[2]

    @principal_point_x.setter
    def principal_point_x(self, value):
        self[2] = value

    @property
    def principal_point_y(self):
        return self[3]

    @principal_point_y.setter
    def principal_point_y(self, value):
        self[3] = value

    def to_matrix(self) -> Tensor:
        return torch.tensor([[self.focal_length_x, 0, self.principal_point_x],
                             [0, self.focal_length_y, self.principal_point_y],
                             [0, 0, 1.]], dtype=torch.float32)

    def scale(self, w_scale: float, h_scale: float) -> 'CameraIntrinsicParameters':
        self.focal_length_x *= w_scale
        self.focal_length_y *= h_scale
        self.principal_point_x *= w_scale
        self.principal_point_y *= h_scale
        return self

class EngineMode(Enum):
    """
    Enum class representing the different types of data splits.

    Attributes:
        TRAIN (int): Represents the training data split.
        VALID (int): Represents the validation data split.
        TEST (int): Represents the test data split.
    """
    TRAIN = 0
    VALID = 1
    TEST = 2
    
class DepthFlowGenerator:
    def __init__(self, cfg: edict):
        self._real_shape = None
        self._cfg = cfg
        self._occlusion_threshold = cfg.dataset.occlusion_threshold
        self._occlusion_kernel = cfg.dataset.occlusion_kernel

    def gen_depth_img(self, uv, depth, index, cam_params: CameraIntrinsicParameters):
        device = uv.device

        depth_image = torch.zeros(
            self._real_shape[:2], device=device, dtype=torch.float
        )
        depth_image += 1000.0
        mask_image = (-1) * torch.ones(
            self._real_shape[:2], device=device, dtype=torch.float
        )
        index = index.float()
        depth_image, mask_image = visibility.depth_image(
            uv,
            depth,
            index,
            depth_image,
            mask_image,
            uv.shape[0],
            self._real_shape[1],
            self._real_shape[0],
        )
        depth_image[depth_image == 1000.0] = 0.0
        mask_image_deocclusion = (-1) * torch.ones(
            self._real_shape[:2], device=device, dtype=torch.float
        )
        depth_image_no_occlusion = torch.zeros_like(depth_image, device=device)
        depth_image_no_occlusion, mask_image_deocclusion = visibility.visibility2(
            depth_image,
            cam_params,
            mask_image,
            depth_image_no_occlusion,
            mask_image_deocclusion,
            depth_image.shape[1],
            depth_image.shape[0],
            self._occlusion_threshold,
            self._occlusion_kernel,
        )
        return (
            depth_image_no_occlusion,
            mask_image_deocclusion.int(),
            mask_image,
            mask_image,
        )

    def flatten_mask(self, mask_deocclusion, range_mask_uv):
        index_deocclusion = torch.where(mask_deocclusion > 0)
        mask_deocclusion = mask_deocclusion[
            index_deocclusion[0][:], index_deocclusion[1][:]
        ]
        mask_flatten = torch.zeros(
            range_mask_uv.shape[0], device=mask_deocclusion.device, dtype=torch.int32
        )
        mask_flatten[mask_deocclusion.cpu().numpy() - 1] = mask_deocclusion
        return mask_flatten

    def crop_data_from_dict(self, data: dict, patch_shape, engine_mode: EngineMode):
        H, W = patch_shape[:2]
        patch_H, patch_W = patch_shape[2:]
        assert (
            patch_H <= H and patch_W <= W
        ), "Patch size should be smaller than the image size"
        if engine_mode == EngineMode.TRAIN:
            x = np.random.randint(0, H - patch_H) if H > patch_H else 0
            y = np.random.randint(0, W - patch_W) if W > patch_W else 0
        else:
            x = (H - patch_H) // 2
            y = (W - patch_W) // 2
        # Unpack Data from Dict
        return {
            key: value[..., x : x + patch_H, y : y + patch_W]
            for key, value in data.items()
        }

    def push(self, data_dict: dict, engine_mode=EngineMode.TRAIN):
        vision_images = data_dict["vision_image"]
        point_clouds = data_dict["point_cloud"]
        camera_intrinsic_parameters = data_dict["camera_intrinsic_parameters"]
        T_errs = data_dict["tr_error"]
        R_errs = data_dict["rot_error"]
        orders = data_dict["order"]
        device = vision_images[0].device

        vision_images_input = []
        depth_images_input = []
        depth_images_fine = []
        flow_images_gt = []
        valid_masks = []

        for idx in range(len(vision_images)):
            # 1 Unpack Data
            vision_image = vision_images[idx].to(device)
            point_cloud_fine = point_clouds[idx].clone().to(device)
            cam_params = camera_intrinsic_parameters[idx]
            order = orders[idx]
            self._real_shape = [
                int(vision_image.shape[1]),
                int(vision_image.shape[2]),
                vision_image.shape[0],
            ]
            # 2 Transform Point Cloud
            transform_fine2coarse = get_transform_from_rotation_translation(
                R_errs[idx].to(device), T_errs[idx].to(device)
            ).squeeze(0)
            point_cloud_coarse = apply_transform_to_points(
                point_cloud_fine, transform_fine2coarse
            )
            # 3 Project Point Cloud
            uv_fine, depth_fine, mask_fine = project_with_mask(
                point_cloud_fine, self._real_shape, cam_params, order
            )  # (2, N_fine), (N_fine), (N)
            uv_fine = uv_fine.t().int().contiguous()  # (N_fine, 2)
            uv_coarse, depth_coarse, mask_coarse = project_with_mask(
                point_cloud_coarse, self._real_shape, cam_params, order
            )  # (2, N_coarse), (N_coarse), (N)
            uv_coarse = uv_coarse.t().int().contiguous()  # (N_coarse, 2)

            # 4 Get Flow Set
            flow_set, mask_flow = get_flow_set_from_2pixel_sets(
                uv_coarse, uv_fine, mask_coarse, mask_fine
            )  # (N_flow, 2), (N_flow) in (N)

            # 5 Get Depth Image
            ## 5.1 Filter flow points in coarse points
            mask_flow_coarse = mask_coarse[mask_flow]  # (N_flow)
            range_mask_uv_coarse = (
                torch.arange(mask_flow_coarse.shape[0]).to(device) + 1
            )  # (1, ..., N_flow)
            uv_coarse_in_flow = uv_coarse[
                mask_flow[mask_coarse], :
            ]  # N_coarse [(N_flow) in (N_coarse)] -> (N_flow, 2)
            depth_coarse_in_flow = depth_coarse[
                mask_flow[mask_coarse]
            ]  # N_coarse [(N_flow) in (N_coarse)] -> (N_flow)

            ## 5.2 Filter flow points in fine points
            mask_flow_fine = mask_fine[mask_flow]  # (N_flow)
            range_mask_uv_fine = (
                torch.arange(mask_flow_fine.shape[0]).to(device) + 1
            )  # (1, ..., N_flow)
            uv_fine_in_flow = uv_fine[
                mask_flow[mask_fine], :
            ]  # N_fine [(N_flow) in (N_fine)] -> (N_flow, 2)
            depth_fine_in_flow = depth_fine[
                mask_flow[mask_fine]
            ]  # N_fine [(N_flow) in (N_fine)] -> (N_flow)

            ## 5.3 Get Deocclusion Mask
            _, mask_deocclusion_coarse, _, _ = self.gen_depth_img(
                uv_coarse_in_flow,
                depth_coarse_in_flow,
                range_mask_uv_coarse,
                cam_params,
            )
            mask_depth_coarse = self.flatten_mask(
                mask_deocclusion_coarse, range_mask_uv_coarse
            )
            _, mask_deocclusion_fine, _, _ = self.gen_depth_img(
                uv_fine_in_flow, depth_fine_in_flow, range_mask_uv_fine, cam_params
            )
            mask_depth_fine = self.flatten_mask(
                mask_deocclusion_fine, range_mask_uv_fine
            )

            ## 5.4 Get Depth Image for Training
            depth_image, _, _, _ = self.gen_depth_img(
                uv_coarse, depth_coarse, mask_coarse[mask_coarse], cam_params
            )
            depth_image /= 100.0
            depth_image = depth_image.unsqueeze(0)
            mask_depth = (mask_depth_coarse > 0) & (mask_depth_fine > 0)

            depth_image_fine, _, _, _ = self.gen_depth_img(
                uv_fine, depth_fine, mask_fine[mask_fine], cam_params
            )
            depth_image_fine /= 100.0
            depth_image_fine = depth_image_fine.unsqueeze(0)

            # 6 Get Flow Image
            flow_image = get_flow_image_from_flow_set(
                flow_set, uv_coarse_in_flow, mask_depth, self._real_shape[:2]
            )

            # 7 Crop Data
            vision_image, depth_image, depth_image_fine, flow_image = (
                self.crop_data_from_dict(
                    dict(
                        image=vision_image,
                        depth=depth_image,
                        depth_fine=depth_image_fine,
                        flow=flow_image,
                    ),
                    vision_image.shape[-2:] + (320, 960),
                    engine_mode,
                ).values()
            )

            valid_i = (flow_image[0].abs() < 1000) & (flow_image[1].abs() < 1000)

            vision_images_input.append(vision_image)
            depth_images_input.append(depth_image)
            depth_images_fine.append(depth_image_fine)
            flow_images_gt.append(flow_image)
            valid_masks.append(valid_i)

        data_dict.update(
            {
                "vision_images_input": torch.stack(vision_images_input),
                "depth_images_input": torch.stack(depth_images_input),
                "flow_images_gt": torch.stack(flow_images_gt),
                "valid_masks": torch.stack(valid_masks),
                "depth_images_fine": torch.stack(depth_images_fine),
            }
        )

        return data_dict

def adjust_points_shape(points, shape="N3"):
    """Adjust the shape of points to (N, 3) or (N, 4).

    Args:
        points (Tensor): The input points tensor.
        shape (str): The desired shape of the points. It can be "N3" or "N4", "3N" or "4N".

    Returns:
        Tensor: The adjusted points tensor.  
    """
    if shape[0] == "N":
        if points.shape[-2] in [3, 4]:
            points = points.transpose(-1, -2)  # (3, N) -> (N, 3), (4, N) -> (N, 4)
    elif shape[1] == "N":
        if points.shape[-1] in [3, 4]:
            points = points.transpose(-1, -2)  # (N, 3) -> (3, N), (N, 4) -> (4, N)
    if "4" in shape:
        if points.shape[-1] == 3:
            points = torch.cat(
                [points, torch.ones(points.shape[:-1] + (1,), device=points.device)], dim=-1
            )
        elif points.shape[-2] == 3:
            points = torch.cat(
                [points, torch.ones(points.shape[:-2] + (1,), device=points.device)], dim=-2
            )
    elif "3" in shape:
        if points.shape[-1] == 4:
            points = points[..., :3]
        elif points.shape[-2] == 4:
            points = points[..., :3, :]
    return points

def adjust_coordinate(xyz: Tensor, order=[1,2,0]) -> Tensor:
    """
    Adjusts the coordinates of a given tensor based on the specified order.

    Args:
        xyz (torch.Tensor): The input tensor containing coordinates. C * N.
        order (list, optional): A list specifying the new order of the coordinates. Defaults to [1, 2, 0, 3].

    Returns:
        torch.Tensor: The tensor with adjusted coordinates.
    """
    return xyz[order, :]

def project_with_mask(points: Tensor, image_size, camera_params: CameraIntrinsicParameters, adjust_coordinate_order=None, front=True) -> tuple[Tensor, Tensor, Tensor]:
    """
    Projects 3D points onto a 2D image plane using intrinsic camera parameters and returns the projected points,
    their depths, and a mask indicating valid points.
    Args:
        points (torch.Tensor): A 3xN tensor representing the 3D points to be projected.
        image_size (tuple): A tuple (width, height) representing the size of the image.
    Returns:
        tuple: A tuple containing:
            - uv (torch.Tensor): A 2xN_front tensor of the projected 2D points.
            - depth (torch.Tensor): A 1D tensor of the depths of the valid points.
            - mask (torch.Tensor): A 1D tensor indicating which points are valid after projection.
    Raises:
        TypeError: If the input points tensor does not have a shape of 3xN.
    """
    assert points.dim() == 2, f"points must be a 2D matrix. but points is {points.dim()}D with shape {points.shape}"
    points = adjust_points_shape(points, "3N") # (3, N)
    if adjust_coordinate_order is not None:
        points = adjust_coordinate(points, adjust_coordinate_order)  # (3, N)
    mask = torch.ones(points.shape[1], dtype=torch.bool, device=points.device) # (N)
    if front:
        mask_front = mask_pixels_with_front(points[2, :]) # (N)
        points = points[:, mask_front] # (3, N_front)
        mask = mask_front # (N)
    uv = torch.zeros((2, points.shape[1]), device=points.device) # (2, N_front)
    uv[0, :] = camera_params.focal_length_x * points[0, :] / points[2, :] + camera_params.principal_point_x
    uv[1, :] = camera_params.focal_length_y * points[1, :] / points[2, :] + camera_params.principal_point_y
    
    mask_vision = mask_pixels_with_vision(uv, (0.1, image_size[1]), (0.1, image_size[0])) # (N_front)
    # generate complete indexes
    index_front = torch.where(mask == True)[0] # (N_front)
    mask[index_front] = mask[index_front] & mask_vision # (N_front) in (N) & (N_front)

    return uv[:, mask_vision], points[2, mask_vision], mask

def mask_pixels_with_image_size(pixels: Tensor, image_w_range: tuple[float, float], image_h_range: tuple[float, float]) -> Tensor:
    """Compute the masks of the pixels which are within the range of an image.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        image_w_range (tuple[float, float]): The range of the image width.
        image_h_range (tuple[float, float]): The range of the image height.

    Returns:
        A BoolTensor of the masks of the pixels in the shape of (..., 2). A pixel is with the image if True.
    """
    masks = torch.logical_and(
        torch.logical_and(torch.ge(pixels[0, ...], image_w_range[0]), torch.lt(pixels[0, ...], image_w_range[1])),
        torch.logical_and(torch.ge(pixels[1, ...], image_h_range[0]), torch.lt(pixels[1, ...], image_h_range[1])),
    )
    return masks

def mask_pixels_with_front(depth: Tensor) -> Tensor:
    """Compute the masks of the pixels which are in the front.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        depth (Tensor): the depth tensor.

    Returns:
        A BoolTensor of the masks of the pixels in the shape of (..., 2). A pixel is in the front if True.
    """
    return torch.ge(depth, 0)

def mask_pixels_with_vision(pixels: Tensor, image_w_range: tuple[float, float], image_h_range: tuple[float, float], depth:Tensor=None, front=False):
    """
    Masks the pixels based on vision information.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        image_w_range (tuple[float, float]): The range of the image width.
        image_h_range (tuple[float, float]): The range of the image height.
        depth (Tensor, optional): The depth tensor. Defaults to None.
        front (bool, optional): Whether to mask pixels in the front. Defaults to False.

    Returns:
        Tensor: The masked pixels tensor.
    """
    masks = mask_pixels_with_image_size(pixels, image_w_range, image_h_range)
    if front:
        assert depth is not None, "depth should be provided when front is True"
        masks = torch.logical_and(masks, mask_pixels_with_front(depth))
    return masks

def deproject(uv, pc_project_uv, camera_params: CameraIntrinsicParameters):
    index = np.argwhere(uv > 0)
    mask = uv > 0
    z = uv[mask]
    camera_params = camera_params.cpu().numpy()
    x = (index[:, 1] - camera_params[2]) * z / camera_params[0]
    y = (index[:, 0] - camera_params[3]) * z / camera_params[1]
    zxy = np.array([z, x, y])
    zxy = torch.tensor(zxy, dtype=torch.float32)
    zxyw = torch.cat([zxy, torch.ones(1, zxy.shape[1], device=zxy.device)])
    zxy = zxyw[:3, :]
    zxy = zxy.cpu().numpy()
    xyz = zxy[[1, 2, 0], :]

    # apply mask to pc_project_uv
    pc_project_u = pc_project_uv[:, :, 0][mask]
    pc_project_v = pc_project_uv[:, :, 1][mask]
    pc_project = np.array([pc_project_v, pc_project_u])
    match_index = np.array([index[:, 0], index[:, 1]])

    return xyz.transpose(), pc_project.transpose(), match_index.transpose()

def get_flow_image_from_flow_set(projected_points, index, mask, shape):
    """
    Computes the optical flow from projected points.

    Args:
        projected_points (torch.Tensor.int): A tensor containing the projected points.
        index (torch.Tensor): A tensor containing the indices of the projected points.
        mask (torch.Tensor): A boolean tensor used to mask the projected points and indices.
        shape (tuple): The shape of the output flow tensor.

    Returns:
        torch.Tensor: A tensor representing the optical flow with shape (2, *shape).
    """
    projected_points_mask = projected_points[mask, :].float()
    index_mask = index[mask, :].t()
    flow = torch.zeros((2, *shape), device=projected_points.device, dtype=torch.float)
    flow[0].index_put_((index_mask[1], index_mask[0]), projected_points_mask[:, 0])
    flow[1].index_put_((index_mask[1], index_mask[0]), projected_points_mask[:, 1])
    return flow

def get_flow_set_from_2pixel_sets(pixel_source, pixel_target, mask_source, mask_target):
    """
    Computes the flow set from two sets of pixels and their corresponding masks.

    Args:
        pixel_source (numpy.ndarray): Source pixel coordinates of shape (N_source, 2).
        pixel_target (numpy.ndarray): Target pixel coordinates of shape (N_target, 2).
        mask_source (numpy.ndarray): Boolean mask for the source pixels of shape (N_source,) in (N) .
        mask_target (numpy.ndarray): Boolean mask for the target pixels of shape (N_target,) in (N) .

    Returns:
        tuple: A tuple containing:
            - flow_set (numpy.ndarray): The computed flow set of shape (N_flow, 2).
            - mask (numpy.ndarray): The combined mask of shape (N_flow,) in (N) .
    """
    mask = mask_source & mask_target # (N_flow) in (N) 
    index_source = mask[mask_source] # (N_flow) in (N_source)
    index_target = mask[mask_target] # (N_flow) in (N_target)
    flow_set = pixel_target[index_target, :] - pixel_source[index_source, :] # (N_flow, 2)
    return flow_set, mask

def apply_transform_to_points(
    points_source: Tensor, transform_source: Tensor, disentangled: bool = False
) -> Tensor:
    # clone to avoid in-place operation
    points = points_source.clone()
    transform = transform_source.clone()

    # pre shape check
    transform = (
        transform[None, :, :] if transform.dim() == 2 else transform
    )  # (B, 4, 4)
    flag = False
    if points.dim() == 2:
        points = points[None, :, :]
        flag = True
    if points.shape[1] == 3 or points.shape[1] == 4:  # (B, 3, N) -> (B, N, 3), (B, 4, N) -> (B, N, 4)
        points = points.transpose(1, 2)

    if disentangled:
        points_mean = points[..., :3].mean(dim=1)[:, None, :]  # (B, 1, 3)
        points[..., :3] -= points_mean  # (B, N, 3)
    rotation = transform[:, :3, :3]  # (B, 3, 3)
    translation = transform[:, None, :3, 3]  # (B, 1, 3)
    points[..., :3] = torch.matmul(points[..., :3], rotation.transpose(-1, -2))
    if disentangled:
        points[..., :3] += points_mean
    points[..., :3] += translation
    if flag:
        points = points.squeeze(0)
    return points

def get_transform_from_rotation_translation(
    rotation: Tensor, translation: Tensor
) -> Tensor:
    """Compose transformation matrix from rotation matrix and translation vector.
    Args:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    Returns:
        transform (Tensor): (*, 4, 4) float
    """
    input_shape = rotation.shape
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)
    transform = torch.eye(4).to(rotation).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.view(*output_shape)
    return transform.float()

def get_rotation_translation_from_transform(transform: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose transformation matrix into rotation matrix and translation vector.
    Args:
        transform (Tensor): (*, 4, 4)
    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


evaluations = {}

def register_evaluation(cls):
    evaluations[cls.__name__] = cls
    return cls

def get_evaluation(name: str, cfg: edict):
    assert name in evaluations, f"evaluation {name} is not registered"
    return evaluations[name](cfg)

models = {}

def register_model(cls):
    models[cls.__name__] = cls
    return cls

def create_model(cfg: edict):
    assert cfg.model.name in models, f"model {cfg.model.name} is not registered"
    return models[cfg.model.name](cfg)

datasets = {}

def register_dataset(cls):
    datasets[cls.__name__] = cls
    return cls

def create_dataset(cfg: edict=None, engine_mode: EngineMode=None):
    assert cfg is not None, 'cfg must be provided to create dataset'
    assert 'dataset' in cfg or 'name' in cfg, 'dataset must be provided to create dataset'
    assert engine_mode is not None, 'engine_mode must be provided to create dataset'
    if 'dataset' in cfg:
        assert 'name' in cfg['dataset'], 'dataset name must be provided to create dataset'
        assert cfg['dataset']['name'] in datasets, f"dataset {cfg['dataset']['name']} is not registered"
        return datasets[cfg['dataset']['name']](cfg, engine_mode)
    else:
        return datasets[cfg['name']](cfg, engine_mode)

def merge_inputs(queries):
    # point_clouds = []
    # imgs = []
    calibs = []
    orders = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
            if key not in ['camera_intrinsic_parameters', 'order']}
    for input in queries:
        # point_clouds.append(input['point_cloud'])
        # imgs.append(input['vision_image'])
        calibs.append(input['camera_intrinsic_parameters'])
        orders.append(input['order'])
    # returns['point_cloud'] = point_clouds
    # returns['vision_image'] = imgs
    returns['camera_intrinsic_parameters'] = calibs
    returns['order'] = orders
    return returns

def get_test_data_loader(cfg: edict) -> Dataset:
    get_logger().info('Loading test data loader...')
    test_dataset = create_dataset(cfg.dataset, engine_mode=EngineMode.TEST)
    test_data_loader = build_dataloader(test_dataset, 
                                        num_workers=cfg.dataset['num_workers'], 
                                        batch_size=1,
                                        shuffle=False,
                                        collate_fn=merge_inputs
                                        )
    return test_data_loader

def reset_seed_worker_init_fn(worker_id):
    """Reset NumPy and Python seed for data loader worker."""
    seed = initial_seed() % (2 ** 32)
    np_random.seed(seed)
    random.seed(seed)

def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    sampler=None,
    pin_memory=True,
    drop_last=False,
):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader

from typing import overload

@overload
def ensure_dir(path: str): ...

@overload
def ensure_dir(path: Path): ...

def ensure_dir(path):
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    else:
        assert path.is_dir(), f"'{path}' already exists but is not a directory."

class MetricsManager:
    def __init__(self):
        self._metrics: dict[str, list[float]] = {}
        self._keys: list[str] = []
    
    def register_metric(self, key: str, metric: float):
        self._metrics[key] = []
        self._keys.append(key)
        self._metrics[key].append(metric)
    
    def update_metric(self, key: str, metric: float):
        if key not in self._keys:
            self.register_metric(key, metric)
            return
        self._metrics[key].append(metric)
    
    def update(self, metric_dict: dict[str, float]):
        for key, metric in metric_dict.items():
            self.update_metric(key, metric)
    
    def get_metric_mean_std(self, key: str, threshold_key: str = None, filter_func = None) -> tuple[float, float]:
        assert key in self._keys, f"Key '{key}' not found."
        filter_metrics = self._metrics
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, [key], filter_func)
        return np.mean(filter_metrics[key]), np.std(filter_metrics[key])

    def get_metric_mean(self, key: str, threshold_key: str = None, filter_func = None) -> float:
        assert key in self._keys, f"Key '{key}' not found."
        filter_metrics = self._metrics
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, [key], filter_func)
        return np.mean(filter_metrics[key])

    def get_metrics_mean_std(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, tuple[float, float]]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        mean_std_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            mean_std_metrics['RR'] = filter_metrics['RR']
            mean_std_metrics['threshold_key'] = threshold_key
            mean_std_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                mean_std_metrics[key] = "NoData"
            else:
                mean_std_metrics[key] = (np.mean(filter_metrics[key]), np.std(filter_metrics[key]))
        return mean_std_metrics

    def get_metrics_mean(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, float]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        mean_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            mean_metrics['RR'] = filter_metrics['RR']
            mean_metrics['threshold_key'] = threshold_key
            mean_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                mean_metrics[key] = "NoData"
            else:
                mean_metrics[key] = np.mean(filter_metrics[key])
        return mean_metrics

    def get_metrics_median(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, float]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        median_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            median_metrics['RR'] = filter_metrics['RR']
            median_metrics['threshold_key'] = threshold_key
            median_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                median_metrics[key] = "NoData"
            else:
                median_metrics[key] = np.median(filter_metrics[key])
        return median_metrics

    def get_metrics_mean_std_median(self, keys: list[str] = None, threshold_key: str = None, filter_func = None) -> dict[str, tuple[float, float, float]]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        filter_metrics = self._metrics
        mean_std_median_metrics = {}
        if threshold_key is not None and filter_func is not None:
            filter_metrics = self.filter_metrics(threshold_key, keys, filter_func)
            mean_std_median_metrics['RR'] = filter_metrics['RR']
            mean_std_median_metrics['threshold_key'] = threshold_key
            mean_std_median_metrics['filter_func'] = filter_func.threshold
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            if np.asarray(filter_metrics[key]).size == 0:
                mean_std_median_metrics[key] = "NoData"
            else:
                mean_std_median_metrics[key] = (np.mean(filter_metrics[key]), np.std(filter_metrics[key]), np.median(filter_metrics[key]))
        return mean_std_median_metrics
        
    def filter_metrics(self, threshold_key: str, keys: list[str] = None, filter_func = None) -> dict[str, np.ndarray]:
        if keys is None:
            keys = self._keys
        assert len(keys) > 0, "At least one key must be provided."
        assert threshold_key in self._keys, f"Threshold key '{threshold_key}' not found."
        assert filter_func is not None, "Filter function must be provided."
        filter_indices = np.where(filter_func(np.array(self._metrics[threshold_key])))[0]
        filter_metrics = {}
        filter_metrics['RR'] = (len(filter_indices) / len(self._metrics[threshold_key])) * 100
        for key in keys:
            assert key in self._keys, f"Key '{key}' not found."
            filter_metrics[key] = np.array(self._metrics[key])[filter_indices]
        return filter_metrics
    
    def clear(self):
        self._metrics = {}
        self._keys = []



def setup_engine(seed=None, cudnn_deterministic=True, debug=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(debug)

def load_state_dict(model, state_dict, strict=False):
    logger = get_logger()

    if isinstance(model, DistributedDataParallel):
        model = model.module

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        logger.warn(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warn(f"Unexpected keys: {unexpected_keys}")

    if strict and (len(missing_keys) != 0 or len(unexpected_keys) != 0):
        raise RuntimeError("The keys in model and state_dict do not match.")

class BaseTester(abc.ABC):
    def __init__(self):
        # parser
        parser = get_default_parser()
        self._args = parser.parse_args()
        self._cudnn_deterministic = self._args.cudnn_deterministic

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."
        
        cfg = get_config()
        self._cfg = cfg
        # logger
        self._log_file = cfg.experiment.log_dir / "test.log"
        self._logger = get_logger(cfg, self._log_file)

        # find checkpoint
        self._checkpoint = self._args.checkpoint
        assert Path(self._checkpoint).exists(), f"Checkpoint not found: {self._checkpoint}"
        
        # metrics manager
        self._metrics_manager = MetricsManager()
        
        # initialize
        torch.cuda.set_device(*cfg.gpus)
        setup_engine(seed=cfg.experiment.seed, cudnn_deterministic=self._cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        # data loader
        self.test_loader = None

    @property
    def args(self):
        return self._args

    @property
    def log_file(self):
        return self._log_file

    def load(self, filename, strict=True):
        self.log('Loading from "{}".'.format(filename))
        state_dict = torch.load(filename, map_location=torch.device("cpu"), weights_only=True)
        assert "model" in state_dict, "No model can be loaded."
        load_state_dict(self.model, state_dict["model"], strict=strict)
        self.log("Model has been loaded.")
        if "metadata" in state_dict:
            epoch = state_dict["metadata"]["epoch"]
            total_steps = state_dict["metadata"]["total_steps"]
            self.log(f"Checkpoint metadata: epoch: {epoch}, total_steps: {total_steps}.")

    def register_model(self, model):
        """Register model."""
        model = model.cuda()
        self.model = model
        message = "Model description:\n" + str(model)
        self.log(message)
        return model

    def register_loader(self, test_loader):
        """Register data loader."""
        self.test_loader = test_loader

    def log(self, message, level="INFO"):
        self._logger.log(message, level=level)

    def write_dict(self, data_dict):
        """Write Wandb event."""
        self._logger.wandb_log(data_dict)
        
    def metrics_clear(self):
        self._metrics_manager.clear()
        
    def metrics_update(self, data_dict):
        self._metrics_manager.update(data_dict)

    def metrics_summary_mean(self):
        return self._metrics_manager.get_metrics_mean()
    
    def metrics_summary_mean_std(self):
        return self._metrics_manager.get_metrics_mean_std()

    def before_test_epoch(self):
        self.metrics_clear()

    def before_test_step(self, iteration, data_dict):
        return data_dict

    @abc.abstractmethod
    def test_step(self, iteration, data_dict) -> dict:
        pass

    @abc.abstractmethod
    def eval_step(self, iteration, data_dict, output_dict) -> dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self, summary_dict):
        pass

    def get_log_string(self, iteration, data_dict, output_dict, result_dict) -> str:
        return get_log_string(result_dict)

    @abc.abstractmethod
    def test_epoch(self):
        pass

    def run(self, strict_loading=True):
        assert self.test_loader is not None
        if self._checkpoint is not None:
            self.load(self._checkpoint, strict=strict_loading)
        else:
            self.log("Checkpoint is not specified.", level="WARNING")
        self.model.eval()
        torch.set_grad_enabled(False)
        self.test_epoch()

class SingleTester(BaseTester, abc.ABC):
    def __init__(self):
        super().__init__()

    def test_epoch(self):
        # before epoch
        self.before_test_epoch()
        # setup watcher
        timer = Timer()
        # test loop
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        timer.tic("data")
        for batch_index, data_dict in pbar:
            # on start
            self.iteration = batch_index + 1
            data_dict = move_to_cuda(data_dict)
            data_dict = self.before_test_step(self.iteration, data_dict)
            timer.toc("data")
            # test step
            torch.cuda.synchronize()
            timer.tic("model")
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.toc("model")
            # eval step
            timer.tic("data")
            timer.tic("eval")
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            timer.toc("eval")
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = tensor_to_array(result_dict)
            self.metrics_update(result_dict)
            message = self.get_log_string(self.iteration, data_dict, output_dict, result_dict)
            pbar.set_description(message + ", " + timer.tostring(keys=["data", "model", 'eval'], verbose=False))
            torch.cuda.empty_cache()
        # summary logging
        summary_dict = self.metrics_summary_mean_std()
        self.write_dict(summary_dict)
        message = get_log_string(summary_dict, time_dict=timer.summary(keys=["data", "model", 'eval']))
        self.log(message, level="SUCCESS")
        # after epoch
        self.after_test_epoch(summary_dict)

import time

class Timer:
    def __init__(self):
        self._total_time = {}
        self._count_time = {}
        self._last_time = {}
        self._keys = []

    def register_timer(self, key):
        self._total_time[key] = 0.0
        self._count_time[key] = 0
        self._last_time[key] = None
        self._keys.append(key)

    def tic(self, key):
        if key not in self._keys:
            self.register_timer(key)
        self._last_time[key] = time.time()

    def toc(self, key):
        assert key in self._keys, f"'{key}' is not registered in {self._keys}. Please register it first."
        assert self._last_time[key] is not None, "'tic' must be called before 'toc'."
        duration = time.time() - self._last_time[key]
        self._total_time[key] += duration
        self._count_time[key] += 1
        self._last_time[key] = None

    def get_time(self, key):
        assert key in self._keys, f"'{key}' is not registered in {self._keys}. Please register it first."
        assert self._count_time[key] > 0, f"'toc' must be called at least once for key '{key}'."
        return self._total_time[key] / self._count_time[key]

    def tostring(self, keys=None, verbose=True):
        if keys is None:
            keys = self._keys
        if verbose:
            log_strings = [f"{key}: {self.get_time(key):.3f}s" for key in keys if key in self._keys]
            format_string = ", ".join(log_strings)
        else:
            log_strings = [f"{self.get_time(key):.3f}s" for key in keys if key in self._keys]
            format_string = "time: " + "/".join(log_strings)
        return format_string

    def summary(self, keys=None):
        if keys is None:
            keys = self._keys
        summary_dict = {key: self.get_time(key) for key in keys}
        return summary_dict

import sys
import warnings
import loguru
import wandb

class Logger:
    """Advanced logger with stderr, log file and Wandb support.
    
    When DistributedDataParallel is enabled, only ERROR logs are activated for slave processes.
    """
    def __init__(self, cfg: edict, log_file=None):
        is_master_node = True
        self._logger = loguru.logger
        self._logger.remove()
        fmt_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level><n>{message}</n></level>"
        log_level = "DEBUG" if is_master_node else "ERROR"
        self._logger.add(sys.stderr, format=fmt_str, colorize=True, level=log_level)
        self._logger.info("Command executed: " + " ".join(sys.argv))
        self._log_file = log_file if is_master_node else None
        if self._log_file is not None:
            fmt_str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            self._logger.add(self._log_file, format=fmt_str, level="INFO")
            self._logger.info(f"Logs are saved to {self._log_file}.")
        wandb.init(
            project = cfg.title,
            dir = cfg.experiment.project_dir,
            name = cfg.experiment.experiment_name,
            config = cfg
        )
        
    @property
    def log_file(self):
        return self._log_file

    def log(self, message, level="INFO"):
        if level not in ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            self._logger.warning(f"Unsupported logging level: {level}. Fallback to INFO.")
            level = "INFO"
        self._logger.log(level, message)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)

    def success(self, message):
        self._logger.success(message)

    def warn(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def critical(self, message):
        self._logger.critical(message)

    def wandb_watch(self, model):
        wandb.watch(model)
    
    def wandb_log(self, data_dict):
        wandb.log(data_dict)

_LOGGER = None

def get_logger(cfg=None, log_file=None):
    """Guarantee only one logger per node is built."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(cfg, log_file=log_file)
    elif log_file is not None:
        log_strings = []
        if log_file is not None:
            log_strings.append(f"log_file={log_file}")
        message = "Logger is already initialized. New parameters (" + ",".join(log_strings) + ") are ignored."
        warnings.warn(message)
    return _LOGGER

def get_print_format(value):
    if isinstance(value, (int, str, tuple)):
        return ""
    if value == 0:
        return ".3f"
    if value < 1e-5:
        return ".3e"
    if value < 1e-2:
        return ".6f"
    return ".3f"

def get_format_strings(result_dict):
    """Get format string for a list of key-value pairs."""
    format_strings = []
    if "metadata" in result_dict:
        # handle special key "metadata"
        format_strings.append(result_dict["metadata"])
    for key, value in result_dict.items():
        if key == "metadata":
            continue
        if isinstance(value, (tuple)):
            format_string = f"{key}: " + "/".join([f"{item:{get_print_format(item)}}" for item in value])
        else:
            format_string = f"{key}: {value:{get_print_format(value)}}"
        format_strings.append(format_string)
    return format_strings

def get_log_string(
    result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, time_dict=None
):
    log_strings = []
    if epoch is not None:
        epoch_string = f"epoch: {epoch}"
        if max_epoch is not None:
            epoch_string += f"/{max_epoch}"
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f"iter: {iteration}"
        if max_iteration is not None:
            iter_string += f"/{max_iteration}"
        log_strings.append(iter_string)
    log_strings += get_format_strings(result_dict)
    if lr is not None:
        log_strings.append("lr: {:.3e}".format(lr))
    if time_dict is not None:
        time_string = "time: " + "/".join([f"{time_dict[key]:.3f}s" for key in time_dict])
        log_strings.append(time_string)
    message = ", ".join(log_strings)
    return message

def move_to_cuda(x):
    """Move all tensors to cuda."""
    if isinstance(x, list):
        x = [move_to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([move_to_cuda(item) for item in x])
    elif isinstance(x, dict):
        x = {key: move_to_cuda(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = x.cuda()
    return x

def tensor_to_array(x):
    """Release all pytorch tensors to item or numpy arrays."""
    if isinstance(x, list):
        x = [tensor_to_array(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([tensor_to_array(item) for item in x])
    elif isinstance(x, dict):
        x = {key: tensor_to_array(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x

import argparse

_PARSER = None

def get_default_parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = argparse.ArgumentParser()
    return _PARSER


def parse_args():
    parser = get_default_parser()
    args = parser.parse_args()
    return args

def add_base_args():
    parser = get_default_parser()
    parser.add_argument("--cfg", type=str, required=True, default='./test.toml', help="config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="load from checkpoint")
    parser.add_argument("--cudnn_deterministic", type=bool, default=True, help="use deterministic method")

def add_trainer_args():
    parser = get_default_parser()
    parser.add_argument_group("trainer", "trainer arguments")
    parser.add_argument("--resume", action="store_true", help="resume training from the latest checkpoint")
    parser.add_argument("--log_steps", type=int, default=100, help="logging steps")
    parser.add_argument("--debug", action="store_true", help="debug mode with grad check")
    parser.add_argument("--detect_anomaly", action="store_true", help="detect anomaly with autograd")
    parser.add_argument("--save_latest_n_models", type=int, default=-1, help="save latest n models")
    parser.add_argument("--watch_model", action="store_true", help="watch model with wandb")

def add_tester_args():
    parser = get_default_parser()
    parser.add_argument_group("tester", "tester arguments")

add_base_args()

from easydict import EasyDict
import tomllib as tml
def read_toml_file(file_name: str) -> EasyDict:
    """
    Read a TOML file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the TOML file.

    Returns:
        dict[str, any]: The contents of the TOML file as a dictionary.

    """
    with open(file_name, 'rb') as toml_file:
        return EasyDict(tml.load(toml_file))

_CONFIG = None

def get_deafult_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG

def get_config():
    return get_deafult_config().cfg

import datetime
import threading

class SingletonType(type):
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance
    
class Config(metaclass=SingletonType):
    def __init__(self):
        self.cfg = read_toml_file(parse_args().cfg)
        self.add_experiment_cfg()
    
    def add_experiment_cfg(self):
        """
        Adds experiment configuration details to the given configuration dictionary.

        Returns:
            edict: The updated configuration dictionary with added experiment details.

        The function performs the following actions:
            - Sets the experiment name to the title from the configuration.
            - Sets the experiment time to the current datetime in the format YYYYMMDD_HHMMSS.
            - Sets the working directory to the parent directory of the given filename.
            - Sets the project directory to a subdirectory named after the title within the working directory.
            - Sets the output directory to a subdirectory named after the experiment name within the working directory.
            - Sets the checkpoint directory to a "checkpoints" subdirectory within the output directory.
            - Sets the log directory to a "logs" subdirectory within the output directory.
            - Ensures that all directories ending with "_dir" exist by creating them if necessary.
        """
        if "experiment" not in self.cfg:
            self.cfg.experiment = edict()
        self.cfg.experiment.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg.experiment.working_dir = Path(parse_args().cfg).resolve().parent.parent # ./working_dir/cfg/cfg.toml
        self.cfg.experiment.project_dir = self.cfg.experiment.working_dir / self.cfg.title
        self.cfg.experiment.name_dir = self.cfg.experiment.project_dir / ("train" if self.cfg.mode == "train" else "test")
        self.cfg.experiment.experiment_name = self.cfg.mode + "_" + self.cfg.experiment.time
        self.cfg.experiment.output_dir = self.cfg.experiment.name_dir / self.cfg.experiment.experiment_name
        self.cfg.experiment.checkpoint_dir = self.cfg.experiment.output_dir / "checkpoints"
        self.cfg.experiment.log_dir = self.cfg.experiment.output_dir / "logs"
        self.cfg.experiment.result_dir = self.cfg.experiment.output_dir / "result"
        for dir in self.cfg.experiment:
            if dir.endswith("_dir"):
                ensure_dir(self.cfg.experiment[dir])

    def __str__(self) -> str:
        return f"Configuration details:\n{self.cfg}"

class Evaluation(nn.Module, abc.ABC):
    def __init__(self, cfg: edict):
        self._cfg = cfg
        super(Evaluation, self).__init__()
        
    @abc.abstractmethod
    def evaluation_fn(self, data_dict: dict, output_dict: dict) -> dict:
        raise NotImplementedError
    
    def forward(self, data_dict: dict, output_dict: dict):
        result_dict = self.evaluation_fn(data_dict, output_dict)
        return result_dict

def inverse_rotation_translation(
    rotation: Tensor, translation: Tensor
) -> tuple[Tensor, Tensor]:
    """Inverse rotation and translation.
    Args:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    Returns:
        inv_rotation (Tensor): (*, 3, 3). float
        inv_translation (Tensor): (*, 3). float
    """
    inv_rotation = rotation.transpose(-1, -2).float()  # (*, 3, 3)
    inv_translation = -torch.matmul(
        inv_rotation, translation.view(-1, 3, 1).float()
    ).squeeze(
        -1
    )  # (*, 3)
    return inv_rotation, inv_translation

class TransformDistanceType(Enum):
    """Distance type for rigid transformations."""

    COMMON = 0
    I2D_LOC = 1

def rotation_matrix_distance(
    rotation_matrix1: Tensor, rotation_matrix2: Tensor
) -> Tensor:
    """Compute the distance between two rotation matrices. The error unit of the calculation is the rotation angle error.
    Args:
        rotation_matrix1 (Tensor): (*, 3, 3)
        rotation_matrix2 (Tensor): (*, 3, 3)
    Returns:
        distance (Tensor): (*)
    """
    rotation_matrix1 = rotation_matrix1.view(-1, 3, 3)
    rotation_matrix2 = rotation_matrix2.view(-1, 3, 3)
    return torch.tensor(
        [
            abs(
                torch.acos(
                    (
                        torch.trace(
                            torch.mm(
                                torch.inverse(rotation_matrix1).view(3, 3),
                                rotation_matrix2.view(3, 3),
                            )
                        )
                        - 1
                    )
                    / 2
                )
            )
            * 180.0
            / np.pi
            for i in range(rotation_matrix1.shape[0])
        ]
    )

def rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """Convert rotation matrix to quaternion.
    Args:
        rotation_matrix (Tensor): (*, 3, 3)
    Returns:
        quaternion (Tensor): (*, 4) 【xyzw】
    """
    rotation_matrix = rotation_matrix.view(-1, 3, 3)
    return torch.tensor(
        np.array(
            [
                R.from_matrix(rotation_matrix[i].detach().cpu().numpy()).as_quat()
                for i in range(rotation_matrix.shape[0])
            ]
        )
    )

def transform_distance(
    transform1: Tensor, transform2: Tensor, flag=TransformDistanceType.I2D_LOC
) -> tuple[Tensor, Tensor]:
    """Compute distance between two rigid transformations.
    Args:
        transform1 (Tensor): (*, 4, 4)
        transform2 (Tensor): (*, 4, 4)
    Returns:
        distance(Tensor, Tensor): rotation distance, translation distance
    """
    match flag:
        case TransformDistanceType.COMMON:
            rotation1, translation1 = get_rotation_translation_from_transform(
                transform1
            )  # (*, 3, 3), (*, 3)
            rotation2, translation2 = get_rotation_translation_from_transform(
                transform2
            )  # (*, 3, 3), (*, 3)
            rotation_distance = rotation_matrix_distance(
                rotation1, rotation2
            )  # (*,)
            translation_distance = torch.norm(
                translation1 - translation2, dim=-1
            )  # (*,)
        case TransformDistanceType.I2D_LOC:
            rotation, translation = get_rotation_translation_from_transform(
                transform=torch.matmul(inverse_transform(transform2), transform1)
            )
            rotation_distance = (
                quaternion_distance(
                    rotation_matrix_to_quaternion(rotation),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                )
                * 180.0
                / torch.pi
            )
            translation_distance = torch.norm(translation) * 100  # (*,)
    return rotation_distance, translation_distance

def inverse_transform(transform: Tensor) -> Tensor:
    """Inverse rigid transform.
    Args:
        transform (Tensor): (*, 4, 4)
    Return:
        inv_transform (Tensor): (*, 4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(
        transform
    )  # (*, 3, 3), (*, 3)
    inv_rotation = rotation.transpose(-1, -2)  # (*, 3, 3)
    inv_translation = -torch.matmul(inv_rotation, translation.unsqueeze(-1)).squeeze(
        -1
    )  # (*, 3)
    inv_transform = get_transform_from_rotation_translation(
        inv_rotation, inv_translation
    )  # (*, 4, 4)
    return inv_transform

def quaternion_inverse(quaternion: Tensor) -> Tensor:
    """Compute the inverse of a quaternion.
    Args:
        quaternion (Tensor): (*, 4)
    Returns:
        quaternion_inv (Tensor): (*, 4)
    """
    quaternion = quaternion.view(-1, 4)
    return torch.tensor(
        np.array(
            [
                R.from_quat(quaternion[i].detach().cpu().numpy()).inv().as_quat()
                for i in range(quaternion.shape[0])
            ]
        )
    )

def quaternion_multiply(quaternion1: Tensor, quaternion2: Tensor) -> Tensor:
    """Compute the multiplication of two quaternions.
    Args:
        quaternion1 (Tensor): (*, 4) 【xyzw】
        quaternion2 (Tensor): (*, 4) 【xyzw】
    Returns:
        quaternion (Tensor): (*, 4) 【xyzw】
    """
    q = quaternion1.view(-1, 4)[:, [3, 0, 1, 2]]
    r = quaternion2.view(-1, 4)[:, [3, 0, 1, 2]]
    t = torch.zeros(q.shape[0], 4, device=q.device)
    t[:, 0] = (
        r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    )
    t[:, 1] = (
        r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    )
    t[:, 2] = (
        r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    )
    t[:, 3] = (
        r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    )
    return t

def quaternion_distance(quaternion1: Tensor, quaternion2: Tensor) -> Tensor:
    """Compute the distance between two quaternions. The error unit of the calculation is the rotation angle error.
    Args:
        quaternion1 (Tensor): (*, 4)
        quaternion2 (Tensor): (*, 4)
    Returns:
        distance (Tensor): (*) 弧度制
    """
    t = quaternion_multiply(quaternion1, quaternion_inverse(quaternion2))
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))

def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix.
    Args:
        quaternion (Tensor): (*, 4)
    Returns:
        rotation_matrix (Tensor): (*, 3, 3) float
    """
    quaternion = quaternion.view(-1, 4)
    return torch.tensor(
        np.array(
            [
                R.from_quat(quaternion[i].detach().cpu().numpy()).as_matrix()
                for i in range(quaternion.shape[0])
            ]
        )
    ).float()
    
def rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """Convert rotation matrix to quaternion.
    Args:
        rotation_matrix (Tensor): (*, 3, 3)
    Returns:
        quaternion (Tensor): (*, 4) 【xyzw】
    """
    rotation_matrix = rotation_matrix.view(-1, 3, 3)
    return torch.tensor(
        np.array(
            [
                R.from_matrix(rotation_matrix[i].detach().cpu().numpy()).as_quat()
                for i in range(rotation_matrix.shape[0])
            ]
        )
    )

def rotation_vector_to_rotation_matrix(rotation_vector: Tensor) -> Tensor:
    """Convert rotation vector to rotation matrix.
    Args:
        rotation_vector (Tensor): (*, 3)
    Returns:
        rotation_matrix (Tensor): (*, 3, 3)
    """
    rotation_vector = torch.tensor(rotation_vector).view(-1, 3)
    return torch.tensor(
        np.array(
            [
                R.from_rotvec(rotation_vector[i].detach().cpu().numpy()).as_matrix()
                for i in range(rotation_vector.shape[0])
            ]
        )
    )

def angle_to_rotation_matrix(angle: Tensor, degrees: bool = True) -> Tensor:
    """Convert angle to rotation matrix.
    Args:
        angle (Tensor): (*, 3)
        degrees (bool): True means degrees, False means radians.
    Returns:
        rotation_matrix (Tensor): (*, 3, 3) float
    """
    angle = angle.view(-1, 3)
    return torch.tensor(
        np.array(
            [
                R.from_euler(
                    "xyz", angle[i].detach().cpu().numpy(), degrees=degrees
                ).as_matrix()
                for i in range(angle.shape[0])
            ]
        ), device=angle.device
    ).float()

def is_path_exist(*file_parts: str) -> bool:
    """
    Check if a file or directory exists by joining multiple path components using pathlib.

    Args:
        *file_parts (str): Multiple components of the file or directory path.

    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    path = Path(*file_parts)
    return path.exists()

def read_image_file(file_name: str) -> Image.Image:
    """
    Load an image from a given file path.

    Args:
        img_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    try:
        img = Image.open(file_name)
        return img
    except IOError as e:
        raise IOError(f"Error opening image file {file_name}: {e}")

def read_h5_file(file_name: str, keys: list[str] | str) -> dict[str, ndarray]:
    """
    Read data from an HDF5 file.

    Args:
        file_name (str): The path to the HDF5 file.
        keys (list[str] | str): The key(s) of the dataset(s) to read from the file. If a single key is provided as a string,
                                it will be converted to a list with a single element.

    Returns:
        dict[str, ndarray]: A dictionary where the keys are the provided dataset keys and the values are the corresponding
                            NumPy arrays containing the data.

    Raises:
        Exception: If there is an error reading the file.

    """
    try:
        keys = [keys] if isinstance(keys, str) else keys
        with h5py.File(file_name, 'r') as hf:
            h5_data = {key: hf[key][:] for key in keys}
        return h5_data
    except Exception as e:
        print(f'File Broken: {file_name}')
        raise e
    
def create_df(data: dict[str, list[any]]) -> pd.DataFrame:
    """
    Create a DataFrame from a dictionary.

    Args:
        data (dict[str, list[any]]): A dictionary where the keys are the column names and the values are lists of data.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the dictionary.

    """
    return pd.DataFrame(data)

def read_csv_file(file_parts: list[str] | str, **kwargs) -> pd.DataFrame:
    """
    Read data from a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv`.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.

    Raises:
        Exception: If there is an error reading the file.

    """
    try:
        file_name = Path(*([file_parts] if isinstance(file_parts, str) else file_parts))
        return pd.read_csv(file_name, **kwargs)
    except Exception as e:
        print(f'File Broken: {file_name}')
        raise e

def write_csv_file(data: pd.DataFrame, file_parts: list[str] | str, **kwargs) -> None:
    """
    Write data to a CSV file.

    Args:
        data (pd.DataFrame): The data to write to the CSV file.
        file_parts (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to `pd.to_csv`.

    """
    file_name = Path(*([file_parts] if isinstance(file_parts, str) else file_parts))
    data.to_csv(file_name, **kwargs)
    
class Dataset_I2P(Dataset, abc.ABC):
    def __init__(self, cfg: edict, engine_mode:EngineMode=EngineMode.TRAIN) -> None:
        super(Dataset_I2P, self).__init__()
        self._cfg = cfg
        self._engine_mode = engine_mode
        self._w_scale = cfg.w_scale
        self._h_scale = cfg.h_scale
        self._adjust_coordinate_order = cfg.adjust_coordinate_order if cfg.adjust_coordinate_order != "" else None
        self.GTs_R = {}
        self.GTs_T = {}
        self.all_files = []
        if engine_mode == EngineMode.TRAIN:
            assert 'train_sequences' in cfg, 'train_sequences must be provided in the configuration'
            for sequence in cfg['train_sequences']:
                self.process_sequence(sequence)
        else:
            self.process_sequence(cfg['test_sequence'])
        self.test_RT = self.get_test_RT()
    
    @abc.abstractmethod
    def process_sequence(self, sequence: str) -> None:
        pass

    def get_test_RT(self) -> list:
        test_RT = []
        if self._engine_mode == EngineMode.TRAIN:
            return test_RT
        test_RT_file = '/'.join([self._cfg['root_folder'], f'test_RT_seq{self._cfg.test_sequence}_{self._cfg.max_r:.2f}_{self._cfg.max_t:.2f}.csv'])
        if not is_path_exist(test_RT_file):
            get_logger().success(f'TEST SET - Not found: {test_RT_file}, Generating a new one')
            rad_factor = np.pi / 180.0
            len_files = len(self.all_files)
            data = {
                'tx': np.random.uniform(-self._cfg['max_t'], self._cfg['max_t'], len_files),
                'ty': np.random.uniform(-self._cfg['max_t'], self._cfg['max_t'], len_files),
                'tz': np.random.uniform(-self._cfg['max_t'], min(self._cfg['max_t'], 1.0), len_files),
                'rx': np.random.uniform(-self._cfg['max_r'], self._cfg['max_r'], len_files) * rad_factor,
                'ry': np.random.uniform(-self._cfg['max_r'], self._cfg['max_r'], len_files) * rad_factor,
                'rz': np.random.uniform(-self._cfg['max_r'], self._cfg['max_r'], len_files) * rad_factor
            }
            write_csv_file(create_df(data), test_RT_file)
        get_logger().success(f'TEST SET: Using this file: {test_RT_file}')
        test_RT.extend(read_csv_file(test_RT_file, sep=',').values.tolist())

        assert len(test_RT) == len(self.all_files), f"Something wrong {len(test_RT)} != {len(self.all_files)}"
        return test_RT

    def custom_transform(self, rgb, img_rotation=0., flip=False): # TODO: Update this function
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if self._engine_mode == EngineMode.TRAIN:
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def adjust_point_cloud(self, pc: np.ndarray) -> torch.Tensor:
        """
        Preprocess a point cloud to ensure it is in a homogeneous coordinate system.

        Args:
            pc (numpy.ndarray): The input point cloud, expected to be of shape (N, 3) or (N, 4) or (3, N) or (4, N).

        Returns:
            torch.Tensor: The preprocessed point cloud in homogeneous coordinates. (N, 4)
        """
        if(isinstance(pc, torch.Tensor)):
            pc_in = pc
        else:
            pc_in = torch.from_numpy(pc.astype(np.float32))
        assert pc_in.dim() == 2, f"PointCloud must be a 2D matrix, but got {pc_in.dim()}D with shape {pc_in.shape}"
        if pc_in.shape[0] == 3 or pc_in.shape[0] == 4: # (3, N) or (4, N) -> (N, 3) or (N, 4)
            pc_in = pc_in.t()
        if pc_in.shape[1] == 3:
            pc_in = torch.cat((pc_in, torch.ones(pc_in.shape[0]).unsqueeze(0)), 1) # (N, 4)
        elif pc_in.shape[1] == 4:
            if torch.all(pc_in[:,3] == 1.):
                pc_in[:,3] = 1.
        else:
            raise TypeError("Wrong PointCloud shape", pc_in.shape)
        return pc_in

    def augment_data(self, image, point_cloud, 
            camera_intrinsic_parameters: CameraIntrinsicParameters, 
            camera_extrinsic_parameters: torch.Tensor) -> tuple:
        """
        Augment data by applying random horizontal mirroring and random rotation.

        Args:
            img_path (str): Path to the image file.
            pc_in (numpy.ndarray): The input point cloud data.
            train_mode (bool, optional): Whether the augmentation is for training. Defaults to True.

        Returns:
            tuple: The augmented image and point cloud.
        """
        image_rotation = 0
        h_mirror = False
        if self._engine_mode == EngineMode.TRAIN:
            # Random horizontal mirroring && Random rotation
            image_rotation = np.random.uniform(-5, 5)
            h_mirror = np.random.rand() > 0.5
        image = self.custom_transform(image, image_rotation, h_mirror)
        # Downsample Point Cloud 204800. N*4 -> 204800 * 4
        if self._engine_mode == EngineMode.TRAIN:
            if h_mirror:
                point_cloud[:, 1] *= -1  
                camera_intrinsic_parameters.principal_point_x = image.shape[2] - camera_intrinsic_parameters.principal_point_x
            R = angle_to_rotation_matrix(torch.tensor([image_rotation, 0, 0]))
            T = torch.tensor([0., 0., 0.]).float()
            transform = inverse_transform(get_transform_from_rotation_translation(R, T))
            point_cloud = apply_transform_to_points(point_cloud, transform) 
        image, camera_intrinsic_parameters = self.scale_image(image, camera_intrinsic_parameters)
        if camera_extrinsic_parameters is not None:
            point_cloud = self.adjust_point_cloud(point_cloud)
            point_cloud = apply_transform_to_points(point_cloud[:, :3], camera_extrinsic_parameters)
        point_cloud = self.adjust_point_cloud(point_cloud)
        return image, point_cloud, camera_intrinsic_parameters

    def generate_random_transforms(self, idx: int=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self._engine_mode == EngineMode.TRAIN:
            R, T = get_rotation_translation_from_transform(generate_random_transforms(self._cfg['max_r'], self._cfg['max_t']))
        else:
            R = angle_to_rotation_matrix(torch.tensor(self.test_RT[idx][4:]), False)
            T = torch.tensor(self.test_RT[idx][1:4])
        return inverse_rotation_translation(R, T)

    def scale_image(self, image, cam_params: CameraIntrinsicParameters):
        self._real_shape = [int(image.shape[1] * self._h_scale), int(image.shape[2] * self._w_scale), image.shape[0]] # H, W, C
        if self._w_scale == 1 and self._h_scale == 1:
            return image, cam_params
        downsample = transforms.Resize(self._real_shape[:2], interpolation=Image.NEAREST)
        image = downsample(image)
        cam_params.scale(self._w_scale, self._h_scale)
        return image, cam_params

    @abc.abstractmethod
    def get_camera_parameters(self, path: str) -> tuple[CameraIntrinsicParameters, torch.Tensor]:
        pass
    
    @abc.abstractmethod
    def get_point_cloud_path(self, idx) -> str:
        pass
    
    @abc.abstractmethod
    def get_image_path(self, idx) -> str:
        pass
    
    @abc.abstractmethod
    def get_camera_parameters_path(self, idx) -> str:
        pass
    
    def __len__(self) -> int:
        return len(self.all_files)
    
    def __getitem__(self, idx):
        image_path = self.get_image_path(idx)
        image = read_image_file(image_path)
        point_cloud_path = self.get_point_cloud_path(idx)
        point_cloud = self.adjust_point_cloud(read_h5_file(point_cloud_path, 'PC')['PC'])
        camera_parameters_path = self.get_camera_parameters_path(idx)
        camera_intrinsic_parameters, camera_extrinsic_parameters = self.get_camera_parameters(camera_parameters_path)
        
        image, point_cloud, camera_intrinsic_parameters = self.augment_data(
            image, point_cloud, camera_intrinsic_parameters, camera_extrinsic_parameters
            )
        R, T = self.generate_random_transforms(idx)
        return {'vision_image': image, 'point_cloud': point_cloud, 'camera_intrinsic_parameters': camera_intrinsic_parameters,
                    'tr_error': T, 'rot_error': R, 'order': self._adjust_coordinate_order}

def generate_random_translation(max_offset: float) -> list[float]:
    """
    Generate a random translation vector within the specified maximum offset.

    Args:
        max_offset (float): The maximum offset for each translation component.

    Returns:
        list[float]: A list containing the randomly generated translation vector [transl_x, transl_y, transl_z].
    """
    transl_x = np.random.uniform(-max_offset, max_offset)
    transl_y = np.random.uniform(-max_offset, max_offset)
    transl_z = np.random.uniform(-max_offset, min(max_offset, 1.0))
    return [transl_x, transl_y, transl_z]

def generate_random_rotation_euler(max_angle: float) -> list[float]:
    """
    Generates a random rotation in Euler angles representation.

    Args:
        max_angle (float): The maximum angle in radians for each Euler angle.

    Returns:
        Tensor: A tensor representing the rotation in Euler angles.

    """
    rotation_euler = [np.random.uniform(-max_angle, max_angle) for _ in range(3)]
    return rotation_euler

def generate_random_transforms(max_angle: float, max_offset: float) -> Tensor:
    """Generate random rotation and translation.
    Args:
        batch_size (int): number of samples
        device (torch.device): device
    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation_euler = generate_random_rotation_euler(max_angle)
    rotation_matrix_tensor = angle_to_rotation_matrix(torch.tensor(rotation_euler))
    translation_tensor = torch.tensor(generate_random_translation(max_offset))
    return get_transform_from_rotation_translation(
        rotation_matrix_tensor, translation_tensor
    )

