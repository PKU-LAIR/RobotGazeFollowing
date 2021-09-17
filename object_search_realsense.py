from object_detector import ObjectDetector
from target_estimator import TargetEstimator
from visualizer import Visualizer

import torch
import torch.backends.cudnn as cudnn
import torchvision
import detectron2
import pyrealsense2 as rs

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


def realsense_get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    return depth_frame, color_frame, depth_intrin


def draw_ids(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        # color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # cv2.rectangle(
        #     img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


class ObjectSearcher():
    def __init__(self, haitai, pipeline, align, depth_scale):
        self.object_detector = ObjectDetector()
        self.target_estimator = TargetEstimator()
        self.visualizer = Visualizer()

        self.haitai = haitai
        self.pipeline = pipeline
        self.align = align
        self.depth_scale = depth_scale

    def search_object(self, position, start_point):
        pitch = position[0] * np.pi / 180
        yaw = position[1] * np.pi / 180

        visualizer.add_attention(start_point, yaw, pitch)

        # TODO: position vector in the camera coordinates
        position_vec = -np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])

        if position_vec[0] > 0:
            rot_direction = 1
        else:
            rot_direction = -1

        init_angle = 180
        rot_step = 2
        target_angle = init_angle

        while True:
            self.haitai.set_abs_angle(target_angle, wait=True)

            depth_frame, color_frame, depth_intrin = realsense_get_frames(self.pipeline, self.align)
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            classes, points, ids = self.object_detector.get_prediction(color_image, depth_image, depth_intrin,
                                                                       self.depth_scale, total_points_num=10)

            curr_angle = self.haitai.get_curr_angle()
            rotation_matrix = get_rotation_matrix(radians(curr_angle - initial_angle))
            rotated_points = points @ rotation_matrix.T

            points_direction = rotated_points - start_point  # (N, 3) array of direction from start to object
            position_vec = position_vec / np.linalg.norm(position_vec)
            points_direction_norm = (points_direction * points_direction).sum(axis=1) ** 0.5
            points_direction = points_direction / points_direction_norm[:, None]
            angles = np.arccos(points_direction @ attention_vec.T) * 180 / np.pi

            self.target_estimator.update_info(rotated_points, classes, angles, ids)

            target_angle = target_angle + rot_direction * rot_step
            self.visualizer.update_renderer()

            if abs(target_angle - init_angle) > 180:
                break

        self.visualizer.add_points(self.target_estimator.all_points, self.target_estimator.all_ids)
        target_class, target_id = self.target_estimator.get_target_estimation()

        return target_class, target_id
