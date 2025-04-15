# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time

import cv2
import numpy as np
import pycocotools.mask as mask_util

from fsdet.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
)

from .colormap import random_color
timer = 0

Rect1 = {}
Rect2 = []
def CountIOU(RecA, RecB):
    xA = max(RecA[0], RecB[0])
    yA = max(RecA[1], RecB[1])
    xB = min(RecA[2], RecB[2])
    yB = min(RecA[3], RecB[3])
    # 计算交集部分面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # 计算预测值和真实值的面积
    RecA_Area = (RecA[2] - RecA[0] + 1) * (RecA[3] - RecA[1] + 1)
    RecB_Area = (RecB[2] - RecB[0] + 1) * (RecB[3] - RecB[1] + 1)
    # 计算IOU
    iou = interArea / float(RecA_Area + RecB_Area - interArea)

    return iou

class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "color", "ttl"]

    def __init__(self, label, bbox, color, ttl):
        self.label = label
        self.bbox = bbox
        self.color = color
        self.ttl = ttl


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self._old_instances = []
        # assert instance_mode in [
        #     ColorMode.IMAGE,
        #     ColorMode.IMAGE_BW,
        # ], "Other mode not supported yet."
        self._instance_mode = instance_mode

    def draw_instance_predictions(self, frame, predictions,frameno):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores".

        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        detected = [
            _DetectedInstance(classes[i], boxes[i], color=None, ttl=8)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)

        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image()
            alpha = 0.3
        else:
            alpha = 0.5
        global Rect1
        for cls in classes:
            # if cls==0:
            #     Rect1 = {k:v for k,v in Rect1.items() if v>-3}
            #     if len(Rect1.keys()) == 0:
            #         for box in boxes:
            #             Rect1[tuple(box)]=0
            #
            #     for rect in list(Rect1.keys()):
            #         i = 0
            #         for box in boxes:
            #             if CountIOU(box, rect) > 0.8:
            #                 i=i+1
            #                 Rect1[rect] = Rect1[rect] + 1
            #             else:
            #                 Rect1[tuple(box)] = 0
            #         if i==0:
            #             Rect1[rect] = 0
            #     pboxes = list({k:v for k,v in Rect1.items() if v>3}.keys())
            #     boxes = pboxes[:]
            #     for i in range(0,len(pboxes)):
            #         for j in range(i+1,len(pboxes)):
            #             if(CountIOU(pboxes[i], pboxes[j]) > 0.7 and (pboxes[j] in boxes)):
            #                 boxes.remove(pboxes[j])
            #     frame_visualizer.overlay_instances(
            #         boxes=boxes,  # boxes are a bit distracting
            #         labels=None,
            #         assigned_colors=None,
            #         alpha=alpha,
            #     )
            #     vis_frame = cv2.cvtColor(frame_visualizer.output.get_image(), cv2.COLOR_RGB2BGR)
            #     savepath = "/home/lixu/Projects/PycharmProjects/FSCE-main/ores"
            #     if len(boxes) > 0:
            #         print(frameno)
            #         id = int(round(time.time() * 1000))
            #         cv2.imwrite(savepath + '/' + str(id) + '.jpg', vis_frame)
            #     return frame_visualizer.output

            if cls==0:
                frame_visualizer.overlay_instances(
                    boxes=boxes,  # boxes are a bit distracting
                    labels=labels,
                    assigned_colors=colors,
                    alpha=alpha,
                )
                vis_frame = cv2.cvtColor(frame_visualizer.output.get_image(), cv2.COLOR_RGB2BGR)
                savepath = "/home/lixu/Projects/PycharmProjects/FSCE-main/ores"
                print(frameno)
                id = int(round(time.time() * 1000))
                # vis_frame = cv2.resize(vis_frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                vis_frame = cv2.resize(vis_frame, (0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(savepath + '/' + str(id) + '.jpg', vis_frame)
            return frame_visualizer.output



    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with boxes:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        boxes_old = [x.bbox for x in self._old_instances]
        boxes_new = [x.bbox for x in instances]
        ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
        threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]
