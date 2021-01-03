import argparse
import time
from pathlib import Path
import numpy as np 
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from fw_video import Video
from pathlib import Path
import fw_utils as u 
from pprint import pprint
from color_explore import get_green_percent_RGB, get_green_percent_BGR


def detect_one_video(video_path):
    opt = get_opt()
    opt.conf_thres = 0.001 
    opt.classes = 11 
    opt.weights = 'weights/yolov5x4.pt'
    opt.source = video_path
    detect(mode='video')

    output_video_path = Path(video_path).with_suffix('.vis.mp4').__str__()
    object_csv_path = = Path(video_path).with_suffix('.csv').__str__()
    detect(save_img=False, object_csv_path, output_video_path, mode='video'):
    


if __name__ == '__main__':

    detect_one_video(video_path='./minot_130_190.mp4')  
