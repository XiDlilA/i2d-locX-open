import os
import torch
import numpy as np
from easydict import EasyDict as edict
from .utils import is_path_exist, CameraIntrinsicParameters, EngineMode, Dataset_I2P, register_dataset


@register_dataset
class Dataset_Sample(Dataset_I2P):
    def __init__(self, cfg: edict, engine_mode: EngineMode = EngineMode.TRAIN) -> None:
        super(Dataset_Sample, self).__init__(cfg, engine_mode)

    def process_sequence(self, sequence: str) -> None:
        for timestamp in [0, 100, 200, 300]:
            timestamp_formatted = f"{timestamp:06d}"
            map_file_path = [
                self._cfg["root_folder"],
                sequence,
                self._cfg["maps_folder"],
                timestamp_formatted + ".h5",
            ]
            img_file_path = [
                self._cfg["root_folder"],
                sequence,
                self._cfg["imgs_folder"],
                timestamp_formatted + ".png",
            ]
            if not (is_path_exist(*map_file_path) and is_path_exist(*img_file_path)):
                continue
            self.all_files.append("/".join([sequence, timestamp_formatted]))

    def get_test_RT(self) -> list:
        test_RT = []
        if self._engine_mode == EngineMode.TRAIN:
            return test_RT
        rad_factor = np.pi / 180.0
        len_files = len(self.all_files)
        data = [
            [
                i,
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
            ]
            for i, (tx, ty, tz, rx, ry, rz) in enumerate(
                zip(
                    np.random.uniform(
                        -self._cfg["max_t"], self._cfg["max_t"], len_files
                    ),
                    np.random.uniform(
                        -self._cfg["max_t"], self._cfg["max_t"], len_files
                    ),
                    np.random.uniform(
                        -self._cfg["max_t"], min(self._cfg["max_t"], 1.0), len_files
                    ),
                    np.random.uniform(
                        -self._cfg["max_r"], self._cfg["max_r"], len_files
                    )
                    * rad_factor,
                    np.random.uniform(
                        -self._cfg["max_r"], self._cfg["max_r"], len_files
                    )
                    * rad_factor,
                    np.random.uniform(
                        -self._cfg["max_r"], self._cfg["max_r"], len_files
                    )
                    * rad_factor,
                )
            )
        ]
        test_RT.extend(data)

        assert len(test_RT) == len(
            self.all_files
        ), f"Something wrong {len(test_RT)} != {len(self.all_files)}"
        return test_RT

    def get_camera_parameters(
        self, path: str
    ) -> tuple[CameraIntrinsicParameters, torch.Tensor]:
        sequence = int(path)
        if sequence == 0:
            camera_intrinsic_parameters = CameraIntrinsicParameters(
                718.856, 718.856, 607.1928, 185.2157
            )
        elif sequence == 3:
            camera_intrinsic_parameters = CameraIntrinsicParameters(
                721.5377, 721.5377, 609.5593, 172.854
            )
        elif sequence in [5, 6, 7, 8, 9, 10]:
            camera_intrinsic_parameters = CameraIntrinsicParameters(
                707.0912, 707.0912, 601.8873, 183.1104
            )
        else:
            raise TypeError("Sequence Not Available")
        return camera_intrinsic_parameters, None

    def get_point_cloud_path(self, idx) -> str:
        item = self.all_files[idx]
        sequence = str(item.split("/")[0])
        timestamp = str(item.split("/")[1])
        pointcloud_path = os.path.join(
            self._cfg["root_folder"],
            sequence,
            self._cfg["maps_folder"],
            timestamp + ".h5",
        )
        return pointcloud_path

    def get_image_path(self, idx) -> str:
        item = self.all_files[idx]
        sequence = str(item.split("/")[0])
        timestamp = str(item.split("/")[1])
        image_path = os.path.join(
            self._cfg["root_folder"],
            sequence,
            self._cfg["imgs_folder"],
            timestamp + ".png",
        )
        return image_path

    def get_camera_parameters_path(self, idx) -> str:
        item = self.all_files[idx]
        sequence = str(item.split("/")[0])
        return sequence
