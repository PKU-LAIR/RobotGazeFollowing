import time
import numpy as np
import cv2
import pyrealsense2 as rs
from matplotlib import pyplot

from probabilistic_model import rnn_method, thres_method
from eth_xgaze import ETH_Xgaze
from gaze_estimation.utils import load_config
from gaze_estimation.gaze_estimator.common import MODEL3D
from object_search_realsense import ObjectSearcher


class GAZE_TRACKER():
    def __init__(self, haitai, pipeline, align, depth_scale):
        self.target_class = None
        self.target_id = None

        self.fix_start = 0
        self.sacc_start = 0
        self.attention_state = []
        self.fix_thres = 2
        self.sacc_thres = 10

        self.ok = False

        self.prop = rnn_method(5)
        config = load_config()
        self.eth_xgaze = ETH_Xgaze(config)

        self.searcher = ObjectSearcher(haitai, pipeline, align, depth_scale)

    def run(self, color_image, depth_image, depth_intrin):
        self.eth_xgaze.visualizer.set_image(color_image.copy())
        faces = self.eth_xgaze.gaze_estimator.detect_faces(color_image)

        if faces:
            target_face = faces[0]
            self.eth_xgaze.gaze_estimator.estimate_gaze(color_image, target_face)
            self.eth_xgaze.draw_visualization(target_face)

            # gaze_v_cam = target_face.gaze_vector
            # gaze_v_cam = gaze_v_cam / np.linalg.norm(gaze_v_cam)
            angles = np.rad2deg(target_face.vector_to_angle(target_face.gaze_vector))  # [pitch, yaw]

            face_center_xy = target_face.landmarks[MODEL3D.NOSE_INDEX]
            face_center_xy = tuple(np.round(face_center_xy).astype(np.int).tolist())
            face_center_x = face_center_xy[0]
            face_center_y = face_center_xy[1]
            head_depth = depth_image[face_center_y, face_center_x] * depth_scale
            face_center = rs.rs2_deproject_pixel_to_point(depth_intrin, [face_center_x, face_center_y], head_depth)
            # face_center = target_face.center

            face_center = np.asarray(face_center)
            self.prop.store(angles, face_center)
            flag, position, velocity = self.prop.analysis()

            self.eth_xgaze.draw_analysis_info(flag, position, target_face)
            cv2.imshow('frame', self.eth_xgaze.visualizer.image)

            # not focusing
            if flag:
                if self.sacc_start == 0:
                    self.sacc_start = time.time()
                if self.fix_start != 0:
                    self.attention_state.append(['focusing', time.time() - self.fix_start])
                    self.fix_start = 0
                if time.time() - self.sacc_start > self.sacc_thres:
                    self.attention_state.append(['not focusing', time.time() - self.sacc_start])
                    # print('why are you wondering')

            # focusing
            else:
                if self.fix_start == 0:
                    self.fix_start = time.time()
                if self.sacc_start != 0:
                    self.attention_state.append(['not focusing', time.time() - self.sacc_start])
                    self.sacc_start = 0
                if time.time() - self.fix_start > self.fix_thres:
                    # print('what are you thinking about')
                    cv2.imshow('gaze result', self.eth_xgaze.visualizer.image)
                    self.attention_state.append(['focusing', time.time() - self.fix_start])
                    print("focusing time statistics:")
                    print(self.attention_state)
                    self.target_class, self.target_id = self.searcher.search_object(position, face_center)
                    self.ok = True


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


if __name__ == '__main__':
    # if platform.platform() == 'Windows':
    #     haitai = HaiTai('COM4')
    # else:
    #     haitai = Haitai()
    # haitai.connect()
    # haitai.set_abs_angle(180, wait=True)
    haitai = None

    pipeline, align, depth_scale = realsense_configration()
    gaze_tracker = GAZE_TRACKER(haitai, pipeline, align, depth_scale)

    while True:
        time_s = time.time()

        depth_frame, color_frame, depth_intrin = realsense_get_frames(pipeline, align)

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        gaze_tracker.run(color_image, depth_image, depth_intrin)

        if cv2.waitKey(33) == ord('q'):
            break
        if gaze_tracker.ok:
            break

    while True:
        gaze_tracker.searcher.visualizer.update_renderer()
        time.sleep(0.05)

    # cv2.destroyAllWindows()
    # haitai.close()
