import time
from matplotlib import cm
from .model import *
from .evaluation import *
from .dataset import *
from .utils import EngineMode, DepthFlowGenerator, get_evaluation, create_model, get_rotation_translation_from_transform, get_transform_from_rotation_translation, ensure_dir, get_test_data_loader, SingleTester

class Tester(SingleTester):
    def __init__(self):
        super().__init__()

        # dataloader
        start_time = time.time()
        data_loader = get_test_data_loader(self._cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.register_loader(data_loader)
        # model
        model = create_model(self._cfg)
        self.register_model(model)
        # evaluator
        self.eval_func = get_evaluation(self._cfg.evaluation.name, self._cfg)
        # preparation
        self.depth_flow_generator = DepthFlowGenerator(self._cfg)

    def overlay_imgs(self, rgb, lidar):
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        rgb = rgb.clone().cpu().permute(1, 2, 0).numpy()
        rgb = rgb * std + mean
        lidar[lidar == 0] = 1000.0
        lidar = -lidar
        lidar = lidar.clone()
        lidar = lidar.unsqueeze(0)
        lidar = lidar.unsqueeze(0)
        lidar = F.max_pool2d(lidar, 3, 1, 1)
        lidar = -lidar
        lidar[lidar == 1000.0] = 0.0
        lidar = lidar[0][0]
        lidar = lidar.cpu().numpy()
        min_d = 0
        max_d = np.max(lidar)
        lidar = ((lidar - min_d) / (max_d - min_d)) * 255
        lidar = lidar.astype(np.uint8)
        lidar_color = cm.jet(lidar)
        lidar_color[:, :, 3] = 0.5
        lidar_color[lidar == 0] = [0, 0, 0, 0]
        blended_img = lidar_color[:, :, :3] * (
            np.expand_dims(lidar_color[:, :, 3], 2)
        ) + rgb * (1.0 - np.expand_dims(lidar_color[:, :, 3], 2))
        blended_img = blended_img.clip(min=0.0, max=1.0)
        blended_img = cv2.cvtColor(
            (blended_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
        )
        return blended_img

    def render(self, iteration, data_dict, image_name, gt=True):
        vision_image = data_dict["vision_images_input"][0]
        depth_image = data_dict["depth_images_input"][0]
        depth_gt = data_dict["depth_images_fine"][0]
        result_dir = self._cfg.experiment.result_dir / f"iter_{iteration}"
        ensure_dir(result_dir)
        cv2.imwrite(
            f"{result_dir}/{image_name}.png",
            self.overlay_imgs(
                vision_image, torch.tensor(depth_image[0, :, :].cpu().numpy())
            ),
        )
        if gt:
            cv2.imwrite(
                f"{result_dir}/{image_name}_gt.png",
                self.overlay_imgs(
                    vision_image, torch.tensor(depth_gt[0, :, :].cpu().numpy())
                ),
            )

    def test_step(self, iteration, data_dict):
        start_time = time.time()
        output_dict = self.model(data_dict, engine_mode=EngineMode.TEST)
        self.log(
            f"{iteration} Model inference time: {time.time() - start_time:.3f}s.",
            level="DEBUG",
        )
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)
        transform_error = get_transform_from_rotation_translation(
            data_dict["rot_error"][0], data_dict["tr_error"][0]
        )
        rotation_distance, translation_distance = transform_distance(
            transform_error,
            torch.eye(4).to(
                transform_error.device),
            flag=TransformDistanceType.I2D_LOC,
        )
        self.log(
            f"Rotation distance: {rotation_distance.item():.3f}° -> {result_dict['Test_Rotation_Error'].item():.3f}° , Translation distance: -> {translation_distance.item():.3f}cm -> {result_dict['Test_Trans_Error'].item():.3f}cm.",
            level="SUCCESS",
        )
        self.render(iteration, data_dict, "vision_image_with_initial")

        transform_delta = torch.matmul(
            get_transform_from_rotation_translation(
                data_dict["rot_error"][0], data_dict["tr_error"][0]
            ),
            torch.linalg.inv(result_dict["predict"]),
        )
        r, t = get_rotation_translation_from_transform(transform_delta)
        data_dict["tr_error"], data_dict["rot_error"] = [t], [r]
        data_dict = self.depth_flow_generator.push(data_dict, EngineMode.TEST)
        self.render(iteration, data_dict, "vision_image_with_predict", gt=False)
        del result_dict["predict"]
        return result_dict

    def before_test_step(self, iteration, data_dict):
        start_time = time.time()
        data_dict = self.depth_flow_generator.push(data_dict, EngineMode.TEST)
        self.log(
            f"{iteration} Depth generation time: {time.time() - start_time:.3f}s.",
            level="DEBUG",
        )
        return data_dict
