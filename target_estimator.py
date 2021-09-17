import numpy as np
from math import radians, sin, cos
from statistics import mode
import sys

YOLACT_PATH = './yolact/'
sys.path.insert(0, YOLACT_PATH)
from data import cfg

sys.path.remove(YOLACT_PATH)


def idx2name(idx):
    '''
    :param idx: index in dataset
    :return: class name (str)
    '''
    return cfg.dataset.class_names[idx]


class TargetEstimator:
    def __init__(self):
        self.all_points = np.array([], dtype=np.float).reshape(0, 3)
        self.all_classes = np.array([], dtype=np.int).reshape(0)
        self.all_angles = np.array([], dtype=np.int).reshape(0)
        self.all_ids = np.array([], dtype=np.int).reshape(0)

    def update_info(self, points, classes, angles, ids):
        self.all_points = np.vstack([self.all_points, points])
        self.all_classes = np.hstack([self.all_classes, classes])
        self.all_angles = np.hstack([self.all_angles, angles])
        self.all_ids = np.hstack([self.all_ids, ids])

    def get_target_estimation(self, k=20):
        all_topk_classes_name = list()
        all_topk_idx = np.argsort(np.asarray(self.all_angles))[:k]
        all_topk_angles = self.all_angles[all_topk_idx]
        all_topk_classes = self.all_classes[all_topk_idx]
        all_topk_ids = self.all_ids[all_topk_idx]
        for cls in all_topk_classes:
            all_topk_classes_name.append(idx2name(cls))

        print("top k angles")
        print(all_topk_angles)

        print("top k classes")
        print(all_topk_classes_name)

        # print("topk classes id")
        # print(all_topk_classes)

        print("top k ids")
        print(all_topk_ids)

        top_class_name = idx2name(mode(all_topk_classes))
        top_id = mode(all_topk_ids)

        return top_class_name, top_id

    def get_target_idx(self, target_id):
        target_idx = np.argwhere(self.all_ids == target_id).reshape(-1)

        return target_idx
