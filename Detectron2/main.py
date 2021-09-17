import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from random import randint


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


if __name__ == '__main__':
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file('configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = 'models/R101FPN.pkl'
    predictor = DefaultPredictor(cfg)

    test_img = cv2.imread('test.png')
    color_image = np.asanyarray(test_img)

    outputs = predictor(color_image)

    classes = outputs['instances'].pred_classes.cpu().detach().numpy()
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    masks = outputs['instances'].pred_masks.cpu().detach().numpy()

    xywh = xyxy_to_xywh(*boxes[0])
    print(xywh)

    h, w, _ = color_image.shape
    v = Visualizer(test_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('result', out.get_image()[:, :, ::-1])

    while True:
        if cv2.waitKey(33) == ord('q'):
            break
