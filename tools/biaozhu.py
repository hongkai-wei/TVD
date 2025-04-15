# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from fsdet.config import get_cfg
from fsdet.data.detection_utils import read_image
from fsdet.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="FsDet demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/opt/FSCE-main/configurations/PASCAL_VOC/split1/1400shot_CL_IoU.yml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file."#,
                        # default = "/home/mxzh/disk/确定抛洒物视频/36.mp4"
                        # default="/media/lixu/c1baf1b2-b8b3-4d68-b0d4-4f6180bcbe26/抛落物视频/高速/沈海高速/沈海高速-场景2/沈海高速&K100+1112&K82+720下行2&20221203143006&2990.mp4"
                        )
    parser.add_argument(
        "--input",
        default="/opt/FSCE-main/yytest/sjj/images",
        # nargs="+",沈海高速&K100+1112&K82+720下行2&20221202133806&5318.mp4
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/opt/FSCE-main/yytest/sjj/biaozhu",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     args = get_parser().parse_args()
#     setup_logger(name="fvcore")
#     logger = setup_logger()
#     logger.info("Arguments: " + str(args))

#     cfg = setup_cfg(args)

#     demo = VisualizationDemo(cfg)

#     if args.input:
#         if len(args.input) == 1:
#             args.input = glob.glob(os.path.expanduser(args.input[0]))
#             assert args.input, "The input path(s) was not found"
#         inputimage = os.listdir(args.input)
#         for img in tqdm.tqdm(inputimage, disable=not args.output):
#             # use PIL, to be consistent with evaluation
#             path = args.input+'/'+img
#             img = read_image(path, format="BGR")
#             start_time = time.time()
#             predictions, visualized_output = demo.run_on_image(img)
#             logger.info(
#                 "{}: {} in {:.2f}s".format(
#                     path,
#                     "detected {} instances".format(len(predictions["instances"]))
#                     if "instances" in predictions
#                     else "finished",
#                     time.time() - start_time,
#                 )
#             )

#             if args.output:
#                 if os.path.isdir(args.output):
#                     assert os.path.isdir(args.output), args.output
#                     out_filename = os.path.join(args.output, os.path.basename(path))
#                 else:
#                     assert len(args.input) == 1, "Please specify a directory with args.output"
#                     out_filename = args.output
#                 visualized_output.save(out_filename)
#             else:
#                 cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#                 cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
#                 if cv2.waitKey(0) == 27:
#                     break  # esc to quit
#     elif args.webcam:
#         assert args.input is None, "Cannot have both --input and --webcam!"
#         cam = cv2.VideoCapture(0)
#         for vis in tqdm.tqdm(demo.run_on_video(cam)):
#             cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#             cv2.imshow(WINDOW_NAME, vis)
#             if cv2.waitKey(1) == 27:
#                 break  # esc to quit
#         cv2.destroyAllWindows()
#     elif args.video_input:
#         video = cv2.VideoCapture(args.video_input)
#         width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         frames_per_second = video.get(cv2.CAP_PROP_FPS)
#         num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#         basename = os.path.basename(args.video_input)

#         if args.output:
#             if os.path.isdir(args.output):
#                 output_fname = os.path.join(args.output, basename)
#                 output_fname = os.path.splitext(output_fname)[0] + ".mkv"
#             else:
#                 output_fname = args.output
#             assert not os.path.isfile(output_fname), output_fname
#             output_file = cv2.VideoWriter(
#                 filename=output_fname,
#                 # some installation of opencv may not support x264 (due to its license),
#                 # you can try other format (e.g. MPEG)
#                 fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
#                 fps=float(frames_per_second),
#                 frameSize=(width, height),
#                 isColor=True,
#             )
#         assert os.path.isfile(args.video_input)
#         # cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
#         for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
#             if args.output:
#                 # pass
#                 output_file.write(vis_frame)
#             # else:
#             #     cv2.imshow(basename, vis_frame)
#             #     if cv2.waitKey(1) == 27:
#             #         break  # esc to quit
#         video.release()
#         if args.output:
#             output_file.release()
#         else:
#             cv2.destroyAllWindows()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        inputimage = os.listdir(args.input)
        for img in tqdm.tqdm(inputimage, disable=not args.output):
            path = os.path.join(args.input, img)
            original_image = read_image(path, format="BGR")
            height, width = original_image.shape[:2]

            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            # 定义输出图像和文本文件的名称
            base_filename = os.path.splitext(os.path.basename(path))[0]
            out_filename = os.path.join(args.output, base_filename + ".jpg")
            txt_out_filename = os.path.join(args.output, base_filename + ".txt")

            # 保存可视化输出
            visualized_output.save(out_filename)

            # 提取 bounding box 信息并保存为文本文件
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
                scores = instances.scores if instances.has("scores") else None
                classes = instances.pred_classes if instances.has("pred_classes") else None

                with open(txt_out_filename, "w") as f:
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            # 将坐标转换为 YOLO 格式
                            x_center = (box[0] + box[2]) / 2 / width
                            y_center = (box[1] + box[3]) / 2 / height
                            box_width = (box[2] - box[0]) / width
                            box_height = (box[3] - box[1]) / height

                            f.write(f"9 {x_center} {y_center} {box_width} {box_height}\n")

#    ... [处理网络摄像头或视频输入的其他代码部分，如果有的话] ...