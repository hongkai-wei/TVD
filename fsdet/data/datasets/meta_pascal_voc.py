# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from fsdet.structures import BoxMode
from fsdet.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_pascal_voc"]

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor'],
}

def load_filtered_voc_instances(
    name: str, dirname: str, split: str, classnames: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"

    Args example:
        name: voc_2007_trainval_all1_1shot
        dirname: VOC2007 / VOC2012
        split: novel_10shot_split_3_trainval
    """
    use_more_base = 'ploidy' in name
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "vocsplit")

        if use_more_base:
            ploidy = name.split('_')[-1]
            split_id = name.split('_')[3][-1]
            split_dir = os.path.join(split_dir, ploidy, 'split{}'.format(split_id))
            shot = name.split('_')[-3].split('shot')[0]
            seed = int(name.split('_')[-2].replace('seed', ''))
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            if "seed" in name:
                shot = name.split('_')[-2].split('shot')[0]
                seed = int(name.split('_seed')[-1])
                split_dir = os.path.join(split_dir, "seed{}".format(seed))
            else:
                shot = name.split('_')[-1].split('shot')[0]

        for cls in classnames:
            with PathManager.open(os.path.join(split_dir,
                    "box_{}shot_{}_train.txt".format(shot, cls))) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [fid.split('/')[-1].split('.png')[0] \
                                for fid in fileids_]
                fileids[cls] = fileids_

            # fileids 确实是进入了了 3ploidy
            # if cls == 'car':
            #     import pdb; pdb.set_trace()
            #     print(os.path.join(split_dir,
            #         "box_{}shot_{}_train.txt".format(shot, cls)))
            #     print(fileids[cls])
    else:
        with PathManager.open(os.path.join(dirname, "ImageSets", "Main",
                                           split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join("datasets", "VOC{}".format(year))
                anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
                jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".png")
                
                # jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [{
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS
                    }]
                    r["annotations"] = instances
                    dicts_.append(r)

            # this make sure that dataset_dicts has *exactly* K-shot
            if use_more_base and cls in PASCAL_VOC_BASE_CATEGORIES[int(split_id)]:
                if len(dicts_) > int(shot) * int(ploidy[0]):
                    dicts_ = np.random.choice(dicts_, int(shot)*int(ploidy[0]), replace=False)
            else:
                if len(dicts_) > int(shot):
                    dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".png")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append({
                    "category_id": classnames.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                })
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_pascal_voc(
    name, metadata, dirname, split, year, keepclasses, sid):
    if keepclasses.startswith('base_novel'):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith('base'):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith('novel'):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name, lambda: load_filtered_voc_instances(
            name, dirname, split, thing_classes)
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes, dirname=dirname, year=year, split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid]
    )
