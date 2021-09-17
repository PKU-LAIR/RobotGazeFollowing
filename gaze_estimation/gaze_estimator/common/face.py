from typing import Optional

import numpy as np

from gaze_estimation.gaze_estimator.common.face_parts import FaceParts
from gaze_estimation.gaze_estimator.common.face_parts import FacePartsName
from gaze_estimation.gaze_estimator.common.eye import Eye


class Face(FaceParts):
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        super().__init__(FacePartsName.FACE)
        self.bbox = bbox
        self.landmarks = landmarks

        self.reye: Eye = Eye(FacePartsName.REYE)
        self.leye: Eye = Eye(FacePartsName.LEYE)

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])
