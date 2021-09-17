import datetime
import logging
import pathlib
from typing import Optional
import cv2
import numpy as np
import yacs.config
import pyrealsense2 as rs

from gaze_estimation import GazeEstimationMethod, GazeEstimator
from gaze_estimation.gaze_estimator.common import (Face, FacePartsName, Visualizer)
from gaze_estimation.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def realsense_configration():
    """
    realsense d455 configration
    :return: realsense pipeline and align configration
    """
    # Create a realsense pipeline
    d455_pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    # pipeline_wrapper = rs.pipeline_wrapper(d455_pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    # device = pipeline_profile.get_device()
    # device_product_line = str(device.get_info(rs.camera_info.product_line))

    # configration of the resolution
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = d455_pipeline.start(config)

    # Getting the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: ", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    d455_align = rs.align(align_to)

    return d455_pipeline, d455_align, depth_scale


def realsense_get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    return depth_frame, color_frame, depth_intrin


class ETH_Xgaze:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def run(self) -> None:

        pipeline, align, depth_scale = realsense_configration()

        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            depth_frame, color_frame, depth_intrin = realsense_get_frames(pipeline, align)
            color_image = np.asanyarray(color_frame.get_data())

            self.visualizer.set_image(color_image.copy())
            faces = self.gaze_estimator.detect_faces(color_image)

            for face in faces:
                self.gaze_estimator.estimate_gaze(color_image, face)
                self._draw_face_bbox(face)
                self._draw_head_pose(face)
                self._draw_landmarks(face)
                self._draw_face_template_model(face)
                self._draw_gaze_vector(face)

            if self.config.demo.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]
            if self.writer:
                self.writer.write(self.visualizer.image)
            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def draw_visualization(self, face):
        self._draw_face_bbox(face)
        self._draw_head_pose(face)
        self._draw_landmarks(face)
        self._draw_face_template_model(face)
        self._draw_gaze_vector(face)

    def draw_analysis_info(self, flag, position, face):
        self._draw_flag(flag)
        if not isinstance(position, int):
            self._draw_position(face, position)

    def _create_capture(self) -> cv2.VideoCapture:
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(self.config.demo.camera_num)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        output_path = self.output_dir / f'{self._create_timestamp()}.{ext}'
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> None:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        # euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        # pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        # logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
        #             f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'ETH-XGaze':
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            # pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            # logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

    def _draw_flag(self, flag) -> None:
        if flag:
            text = "not focusing"
            color = (255, 0, 0)
        else:
            text = "focusing"
            color = (0, 255, 0)
        cv2.putText(self.visualizer.image, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)

    def _draw_position(self, face, position) -> None:
        length = self.config.demo.gaze_visualization_length
        pitch = position[0] * np.pi / 180
        yaw = position[1] * np.pi / 180
        position_vec = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])
        if self.config.mode == 'ETH-XGaze':
            self.visualizer.draw_3d_line(
                face.center, face.center + length * position_vec, (255, 0, 0))
        else:
            raise ValueError


def main():
    config = load_config()
    eth_xgaze = ETH_Xgaze(config)
    eth_xgaze.run()


if __name__ == '__main__':
    main()
