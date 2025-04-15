import argparse
import copy
import os
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np
from fvcore.common.file_io import PathManager

# VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#                'tvmonitor']  # fmt: skip
VOC_CLASSES = ['litter']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 20], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data = []
    data_per_cat = {c: [] for c in VOC_CLASSES}
    for year in [2007]:
        data_file = "/home/lixu/Projects/PycharmProjects/FSCE-main/datasets/VOC{}/ImageSets/Main/trainval.txt".format(year)
        with PathManager.open(data_file) as f:
            fileids = np.loadtxt(f, dtype=np.str).tolist()
        data.extend(fileids)
    for fileid in data:
        year = "2012" if "_" in fileid else "2007"
        dirname = os.path.join("/home/lixu/Projects/PycharmProjects/FSCE-main/datasets", "VOC{}".format(year))
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)
            print(cls)
            print(anno_file)
        for cls in set(clses):
            data_per_cat[cls].append(anno_file)

    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5, 7,10,1400]
    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in data_per_cat.keys():
            c_data = []
            for j, shot in enumerate(shots):
                diff_shot = shots[j] - shots[j - 1] if j != 0 else 1
                shots_c = random.sample(data_per_cat[c], diff_shot)
                num_objs = 0
                for s in shots_c:
                    if s not in c_data:
                        tree = ET.parse(s)
                        file = tree.find("filename").text
                        year = tree.find("folder").text
                        name = "/home/lixu/Projects/PycharmProjects/FSCE-main/datasets/{}/JPEGImages/{}".format(year, file)
                        c_data.append(name)
                        for obj in tree.findall("object"):
                            if obj.find("name").text == c:
                                num_objs += 1
                        if num_objs >= diff_shot:
                            break
                result[c][shot] = copy.deepcopy(c_data)
        if i == 0:
            save_path = '/home/lixu/Projects/PycharmProjects/FSCE-main/datasets/vocsplit'
        else:
            save_path = "/home/lixu/Projects/PycharmProjects/FSCE-main/datasets/vocsplit/seed{}".format(i)
        os.makedirs(save_path, exist_ok=True)
        for c in result.keys():
            for shot in result[c].keys():
                filename = "box_{}shot_{}_train.txt".format(shot, c)
                with open(os.path.join(save_path, filename), "w") as fp:
                    fp.write("\n".join(result[c][shot]) + "\n")


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)