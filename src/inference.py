"""
This file contains functions to run the inference pipeline including, table detection
table structure extraction, tabular data extraction as well as the final data extraction step.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

from collections import OrderedDict, defaultdict
import json
import argparse
import sys
import xml.etree.ElementTree as ET
import os
import yaml
import openai
from os import listdir
from os.path import join, split
#from spellchecker import SpellChecker
import random
from scipy import stats
import csv
from io import StringIO

import math
import time
import cv2


import torch
import pytesseract
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from fitz import Rect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from pdf2image import convert_from_path
from finalstep import Datasheet

from Microsoft.main import get_model
import Microsoft.postprocess as postprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append('../')
from detr.models import build_model

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6,
            'table row header': 7,
            'table projected column header': 8,
            'table name': 9
        }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map

detection_class_thresholds = {
    "table": 0.8,
    "table rotated": 0.7,
    "no object": 10
}

structure_class_thresholds = {
    "table": 0.8,
    "table column": 0.6,
    "table row": 0.6,
    "table column header": 0.6,
    "table projected row header": 0.3,
    "table spanning cell": 0.4,
    "no object": 10,
    "table row header": 0.6,
    "table projected column header": 0.3,
    "table name": 0.6
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir',
                        help="Directory for input images")
    parser.add_argument('--words_dir',
                        help="Directory for input words")
    parser.add_argument('--out_dir',
                        help="Output directory")
    parser.add_argument('--model_dir',
                        help="Directory containing the models")
    parser.add_argument('--mode',
                        help="The processing to apply to the input image and tokens",
                        choices=['detect', 'recognize', 'extract'])
    parser.add_argument('--structure_config_path',
                        help="Filepath to the structure model config file")
    parser.add_argument('--structure_model_path', help="The path to the structure model")
    parser.add_argument('--detection_config_path',
                        help="Filepath to the detection model config file")
    parser.add_argument('--detection_model_path', help="The path to the detection model")
    parser.add_argument('--detection_device', default="cuda")
    parser.add_argument('--structure_device', default="cuda")
    parser.add_argument('--crops', '-p', action='store_true',
                        help='Output cropped data from table detections')
    parser.add_argument('--objects', '-o', action='store_true',
                        help='Output objects')
    parser.add_argument('--cells', '-l', action='store_true',
                        help='Output cells list')
    parser.add_argument('--html', '-m', action='store_true',
                        help='Output HTML')
    parser.add_argument('--csv', '-c', action='store_true',
                        help='Output CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--visualize', '-z', action='store_true',
                        help='Visualize output')
    parser.add_argument('--crop_padding', type=int, default=10,
                        help="The amount of padding to add around a detected table when cropping.")

    return parser.parse_args()


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    """
        This function converts bounding boxes from center (cx, cy) and dimensions
        (width, height) format to corner coordinates (x_min, y_min, x_max, y_max) format.

        Parameters:
            x: a torch tensor of shape (N, 4), where each row represents a bounding box in
               the format (cx, cy, width, height).

        Returns:
            torch.Tensor: a tensor of shape (N, 4), where each row represents a bounding box in
                          the format (x_min, y_min, x_max, y_max).

        Example:
            >>> boxes = tensor([[0.7144, 0.7041, 0.4671, 0.1790],
                    [0.7242, 0.8372, 0.4353, 0.0722],
                    [0.7185, 0.2427, 0.4608, 0.1866],
                    [0.2639, 0.8494, 0.4279, 0.1019],
                    [0.7231, 0.2556, 0.4637, 0.1933],
                    [0.7020, 0.8345, 0.4775, 0.0771],
                    [0.7228, 0.4020, 0.4662, 0.0838],
                    [0.2587, 0.8531, 0.4175, 0.1005],
                    [0.7288, 0.2515, 0.4436, 0.1952],
                    [0.7187, 0.4130, 0.4697, 0.0883],
                    [0.2520, 0.7219, 0.3947, 0.1001],
                    [0.7204, 0.6813, 0.4551, 0.2093],
                    [0.7178, 0.5298, 0.4666, 0.1485],
                    [0.7173, 0.2512, 0.4550, 0.2144],
                    [0.7182, 0.8302, 0.4473, 0.0782]])
            >>> box_cxcywh_to_xyxy(boxes)
            tensor([[0.4808, 0.6146, 0.9479, 0.7936],
                [0.5066, 0.8011, 0.9418, 0.8733],
                [0.4881, 0.1494, 0.9489, 0.3361],
                [0.0500, 0.7985, 0.4779, 0.9004],
                [0.4913, 0.1589, 0.9550, 0.3522],
                [0.4633, 0.7959, 0.9408, 0.8730],
                [0.4897, 0.3601, 0.9559, 0.4439],
                [0.0500, 0.8028, 0.4674, 0.9033],
                [0.5070, 0.1539, 0.9506, 0.3491],
                [0.4838, 0.3689, 0.9535, 0.4572],
                [0.0547, 0.6719, 0.4493, 0.7720],
                [0.4928, 0.5767, 0.9480, 0.7860],
                [0.4845, 0.4555, 0.9511, 0.6040],
                [0.4898, 0.1440, 0.9448, 0.3584],
                [0.4946, 0.7911, 0.9419, 0.8692]])
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    output = torch.stack(b, dim=1)
    return output


def rescale_bboxes(out_bbox, size):
    """
        Rescales bounding box coordinates from a normalized format (values between 0 and 1)
        to the pixel coordinates of an image of given size.

        Parameters:
            out_bbox (torch.Tensor): A tensor of shape (N, 4) where each row contains
                                     bounding boxes in the format (cx, cy, w, h),
                                     normalized to the range [0, 1].
            size (tuple): A tuple (img_w, img_h) representing the width and height of the image.

        Returns:
            torch.Tensor: A tensor of shape (N, 4) where each row contains bounding boxes
                          in pixel coordinates in the format (x_min, y_min, x_max, y_max).

        Example:
            >>> out_bbox = tensor([[0.5114, 0.7761, 0.9362, 0.5051],
                    [0.5105, 0.5484, 0.9395, 0.8195],
                    [0.5095, 0.8296, 0.9420, 0.3785],
                    [0.5082, 0.5310, 0.9373, 0.7870],
                    [0.4553, 0.7802, 0.7464, 0.1282],
                    [0.2480, 0.0215, 0.3985, 0.0415],
                    [0.5111, 0.8364, 0.9469, 0.0762],
                    [0.5060, 0.0380, 0.9434, 0.0473],
                    [0.2332, 0.0171, 0.3913, 0.0360],
                    [0.5078, 0.0414, 0.9434, 0.0620],
                    [0.5117, 0.3512, 0.9423, 0.0759],
                    [0.5113, 0.8842, 0.9539, 0.0894],
                    [0.3569, 0.5084, 0.4589, 0.7726],
                    [0.5084, 0.5262, 0.9526, 0.7974],
                    [0.5107, 0.5188, 0.9446, 0.8496],
                    [0.5104, 0.0623, 0.9467, 0.0824],
                    [0.2245, 0.5443, 0.3324, 0.7883],
                    [0.5108, 0.9107, 0.9489, 0.0850],
                    [0.5063, 0.0401, 0.9388, 0.0685],
                    [0.5110, 0.2673, 0.9462, 0.1028],
                    [0.4864, 0.0545, 0.9228, 0.0822],
                    [0.4784, 0.0264, 0.9346, 0.0452],
                    [0.5196, 0.7616, 0.8194, 0.0936],
                    [0.5098, 0.2075, 0.9325, 0.3231],
                    [0.5092, 0.5229, 0.9436, 0.8237],
                    [0.3773, 0.4999, 0.6325, 0.8618],
                    [0.2207, 0.5380, 0.3585, 0.8358],
                    [0.5116, 0.2721, 0.9412, 0.0759],
                    [0.2402, 0.5341, 0.3713, 0.8526],
                    [0.5086, 0.2994, 0.9447, 0.6008],
                    [0.5114, 0.1922, 0.9408, 0.0727],
                    [0.3804, 0.5326, 0.5527, 0.7857],
                    [0.5066, 0.0991, 0.9455, 0.1006],
                    [0.5082, 0.0400, 0.9355, 0.0729],
                    [0.5124, 0.4346, 0.9458, 0.0776],
                    [0.2462, 0.5620, 0.4006, 0.8085],
                    [0.4794, 0.5626, 0.7246, 0.8115],
                    [0.5128, 0.0475, 0.9347, 0.0650],
                    [0.4318, 0.4524, 0.7356, 0.8075],
                    [0.5123, 0.5137, 0.9467, 0.0742],
                    [0.4946, 0.0311, 0.9355, 0.0533],
                    [0.5076, 0.0400, 0.9424, 0.0673],
                    [0.5139, 0.0989, 0.9327, 0.1032],
                    [0.3488, 0.0822, 0.5600, 0.1156],
                    [0.5086, 0.6854, 0.9324, 0.7438],
                    [0.5111, 0.7553, 0.9456, 0.0786],
                    [0.2323, 0.5300, 0.2953, 0.8834],
                    [0.4974, 0.8785, 0.7402, 0.1999],
                    [0.5063, 0.9190, 0.8240, 0.0927],
                    [0.5113, 0.6940, 0.7999, 0.5058],
                    [0.4988, 0.8383, 0.8291, 0.1614],
                    [0.5117, 0.6403, 0.9488, 0.3048],
                    [0.5124, 0.6695, 0.9455, 0.6898],
                    [0.5119, 0.2210, 0.9449, 0.1935],
                    [0.3138, 0.5183, 0.4129, 0.7926],
                    [0.2193, 0.5369, 0.3553, 0.8269],
                    [0.5065, 0.0512, 0.9441, 0.0803],
                    [0.3529, 0.5498, 0.4961, 0.8419],
                    [0.4628, 0.0247, 0.9008, 0.0480],
                    [0.5101, 0.9322, 0.9509, 0.1238],
                    [0.5099, 0.0412, 0.9432, 0.0602],
                    [0.5120, 0.0726, 0.9347, 0.0865],
                    [0.5058, 0.5160, 0.9497, 0.8364],
                    [0.5029, 0.5314, 0.9273, 0.8112],
                    [0.2283, 0.5463, 0.3662, 0.8268],
                    [0.5077, 0.2571, 0.9511, 0.4606],
                    [0.5111, 0.1006, 0.9419, 0.1014],
                    [0.2211, 0.5154, 0.2973, 0.7343],
                    [0.5090, 0.5307, 0.9594, 0.8219],
                    [0.5098, 0.5932, 0.9211, 0.1320],
                    [0.5075, 0.5140, 0.9517, 0.8567],
                    [0.4874, 0.5933, 0.8912, 0.1503],
                    [0.5087, 0.5434, 0.9529, 0.8252],
                    [0.5087, 0.3714, 0.9404, 0.7635],
                    [0.2473, 0.0226, 0.4231, 0.0483],
                    [0.4896, 0.0387, 0.9154, 0.0651],
                    [0.2176, 0.5615, 0.3501, 0.7587],
                    [0.5011, 0.0281, 0.9366, 0.0502],
                    [0.4670, 0.0231, 0.9177, 0.0447],
                    [0.7051, 0.5603, 0.5435, 0.8096],
                    [0.5095, 0.8679, 0.9527, 0.1153],
                    [0.4964, 0.5501, 0.7089, 0.6862],
                    [0.5064, 0.0637, 0.9360, 0.0902],
                    [0.5108, 0.9432, 0.9513, 0.1075],
                    [0.5110, 0.9164, 0.9482, 0.0764],
                    [0.5114, 0.1000, 0.9393, 0.0974],
                    [0.3601, 0.0735, 0.6016, 0.1013],
                    [0.5083, 0.3137, 0.9467, 0.0903],
                    [0.3995, 0.4666, 0.5820, 0.7661],
                    [0.5088, 0.1016, 0.9375, 0.1055],
                    [0.4978, 0.8928, 0.8321, 0.2043],
                    [0.5133, 0.8639, 0.9535, 0.0884],
                    [0.5127, 0.6808, 0.9490, 0.5696],
                    [0.3925, 0.1383, 0.6643, 0.0781],
                    [0.2429, 0.5644, 0.3939, 0.8094],
                    [0.5081, 0.5442, 0.9601, 0.8069],
                    [0.5112, 0.6739, 0.9449, 0.0779],
                    [0.4750, 0.0291, 0.8360, 0.0529],
                    [0.5071, 0.0416, 0.9367, 0.0748],
                    [0.4749, 0.0320, 0.8021, 0.0640],
                    [0.5103, 0.0344, 0.9359, 0.0533],
                    [0.4848, 0.0824, 0.8047, 0.0666],
                    [0.2819, 0.1320, 0.4670, 0.1052],
                    [0.2265, 0.5388, 0.3451, 0.8194],
                    [0.3490, 0.0161, 0.5670, 0.0358],
                    [0.5040, 0.4089, 0.8902, 0.7174],
                    [0.5141, 0.9148, 0.9502, 0.1186],
                    [0.2206, 0.5622, 0.3537, 0.7709],
                    [0.3248, 0.5317, 0.4805, 0.8277],
                    [0.2434, 0.5580, 0.4044, 0.7970],
                    [0.5107, 0.5978, 0.9440, 0.7478],
                    [0.5075, 0.0486, 0.9422, 0.0814],
                    [0.4975, 0.0515, 0.9238, 0.0716],
                    [0.1818, 0.0121, 0.2862, 0.0343],
                    [0.5067, 0.0332, 0.9406, 0.0581],
                    [0.5051, 0.5404, 0.9209, 0.5081],
                    [0.5064, 0.0457, 0.9479, 0.0798],
                    [0.5053, 0.0398, 0.9331, 0.0665],
                    [0.5114, 0.5924, 0.9449, 0.0774],
                    [0.3390, 0.9079, 0.5536, 0.1176],
                    [0.3802, 0.4851, 0.5988, 0.8211],
                    [0.5093, 0.0334, 0.9307, 0.0576],
                    [0.5121, 0.5068, 0.9622, 0.9386],
                    [0.2940, 0.4947, 0.4308, 0.8525],
                    [0.5091, 0.0321, 0.9361, 0.0503]])
            >>> size = (1173, 625)
            >>> rescale_bboxes(out_bbox, size)
            tensor([[ 5.0797e+01,  3.2723e+02,  1.1490e+03,  6.4291e+02],
                [ 4.7860e+01,  8.6644e+01,  1.1499e+03,  5.9886e+02],
                [ 4.5156e+01,  4.0023e+02,  1.1502e+03,  6.3680e+02],
                [ 4.6384e+01,  8.5912e+01,  1.1458e+03,  5.7780e+02],
                [ 9.6273e+01,  4.4760e+02,  9.7175e+02,  5.2770e+02],
                [ 5.7184e+01,  4.4483e-01,  5.2457e+02,  2.6375e+01],
                [ 4.4136e+01,  4.9893e+02,  1.1549e+03,  5.4659e+02],
                [ 4.0239e+01,  8.9659e+00,  1.1468e+03,  3.8507e+01],
                [ 4.4112e+01, -5.6050e-01,  5.0306e+02,  2.1919e+01],
                [ 4.2414e+01,  6.5061e+00,  1.1490e+03,  4.5239e+01],
                [ 4.7517e+01,  1.9576e+02,  1.1529e+03,  2.4320e+02],
                [ 4.0269e+01,  5.2473e+02,  1.1592e+03,  5.8058e+02],
                [ 1.4952e+02,  7.6306e+01,  6.8783e+02,  5.5919e+02],
                [ 3.7679e+01,  7.9707e+01,  1.1551e+03,  5.7808e+02],
                [ 4.5043e+01,  5.8715e+01,  1.1531e+03,  5.8974e+02],
                [ 4.3450e+01,  1.3180e+01,  1.1539e+03,  6.4692e+01],
                [ 6.8358e+01,  9.3822e+01,  4.5832e+02,  5.8654e+02],
                [ 4.2636e+01,  5.4265e+02,  1.1557e+03,  5.9577e+02],
                [ 4.3228e+01,  3.6713e+00,  1.1445e+03,  4.6498e+01],
                [ 4.4375e+01,  1.3494e+02,  1.1543e+03,  1.9920e+02],
                [ 2.9329e+01,  8.3594e+00,  1.1118e+03,  5.9731e+01],
                [ 1.3012e+01,  2.3646e+00,  1.1093e+03,  3.0615e+01],
                [ 1.2894e+02,  4.4676e+02,  1.0901e+03,  5.0524e+02],
                [ 5.1089e+01,  2.8709e+01,  1.1449e+03,  2.3067e+02],
                [ 4.3859e+01,  6.9398e+01,  1.1507e+03,  5.8424e+02],
                [ 7.1625e+01,  4.3140e+01,  8.1355e+02,  5.8177e+02],
                [ 4.8617e+01,  7.5068e+01,  4.6912e+02,  5.9742e+02],
                [ 4.8057e+01,  1.4630e+02,  1.1521e+03,  1.9376e+02],
                [ 6.3998e+01,  6.7398e+01,  4.9957e+02,  6.0028e+02],
                [ 4.2568e+01, -6.1844e-01,  1.1507e+03,  3.7488e+02],
                [ 4.8014e+01,  9.7386e+01,  1.1516e+03,  1.4282e+02],
                [ 1.2203e+02,  8.7339e+01,  7.7032e+02,  5.7841e+02],
                [ 3.9700e+01,  3.0516e+01,  1.1488e+03,  9.3412e+01],
                [ 4.7443e+01,  2.2288e+00,  1.1447e+03,  4.7810e+01],
                [ 4.6321e+01,  2.4737e+02,  1.1557e+03,  2.9590e+02],
                [ 5.3844e+01,  9.8583e+01,  5.2378e+02,  6.0391e+02],
                [ 1.3731e+02,  9.8022e+01,  9.8732e+02,  6.0518e+02],
                [ 5.3346e+01,  9.3826e+00,  1.1498e+03,  4.9977e+01],
                [ 7.4993e+01,  3.0430e+01,  9.3790e+02,  5.3511e+02],
                [ 4.5666e+01,  2.9785e+02,  1.1562e+03,  3.4425e+02],
                [ 3.1409e+01,  2.7534e+00,  1.1288e+03,  3.6060e+01],
                [ 4.2736e+01,  3.9592e+00,  1.1482e+03,  4.6049e+01],
                [ 5.5769e+01,  2.9556e+01,  1.1498e+03,  9.4051e+01],
                [ 8.0644e+01,  1.5268e+01,  7.3758e+02,  8.7495e+01],
                [ 4.9728e+01,  1.9593e+02,  1.1435e+03,  6.6079e+02],
                [ 4.4933e+01,  4.4751e+02,  1.1541e+03,  4.9662e+02],
                [ 9.9229e+01,  5.5175e+01,  4.4564e+02,  6.0733e+02],
                [ 1.4927e+02,  4.8660e+02,  1.0176e+03,  6.1151e+02],
                [ 1.1057e+02,  5.4538e+02,  1.0772e+03,  6.0333e+02],
                [ 1.3066e+02,  2.7570e+02,  1.0689e+03,  5.9182e+02],
                [ 9.8815e+01,  4.7351e+02,  1.0713e+03,  5.7438e+02],
                [ 4.3825e+01,  3.0491e+02,  1.1567e+03,  4.9541e+02],
                [ 4.6534e+01,  2.0290e+02,  1.1557e+03,  6.3402e+02],
                [ 4.6264e+01,  7.7686e+01,  1.1546e+03,  1.9861e+02],
                [ 1.2590e+02,  7.6254e+01,  6.1024e+02,  5.7161e+02],
                [ 4.8831e+01,  7.7146e+01,  4.6554e+02,  5.9394e+02],
                [ 4.0416e+01,  6.9505e+00,  1.1478e+03,  5.7109e+01],
                [ 1.2297e+02,  8.0538e+01,  7.0488e+02,  6.0675e+02],
                [ 1.4526e+01,  4.3840e-01,  1.0711e+03,  3.0463e+01],
                [ 4.0636e+01,  5.4392e+02,  1.1560e+03,  6.2127e+02],
                [ 4.4910e+01,  6.9240e+00,  1.1512e+03,  4.4572e+01],
                [ 5.2331e+01,  1.8385e+01,  1.1488e+03,  7.2417e+01],
                [ 3.6320e+01,  6.1101e+01,  1.1503e+03,  5.8386e+02],
                [ 4.6084e+01,  7.8604e+01,  1.1338e+03,  5.8561e+02],
                [ 5.3029e+01,  8.3088e+01,  4.8256e+02,  5.9983e+02],
                [ 3.7742e+01,  1.6763e+01,  1.1534e+03,  3.0464e+02],
                [ 4.7107e+01,  3.1215e+01,  1.1520e+03,  9.4576e+01],
                [ 8.4973e+01,  9.2678e+01,  4.3371e+02,  5.5163e+02],
                [ 3.4408e+01,  7.4857e+01,  1.1598e+03,  5.8852e+02],
                [ 5.7765e+01,  3.2949e+02,  1.1382e+03,  4.1201e+02],
                [ 3.7113e+01,  5.3543e+01,  1.1535e+03,  5.8899e+02],
                [ 4.9059e+01,  3.2384e+02,  1.0944e+03,  4.1779e+02],
                [ 3.7804e+01,  8.1749e+01,  1.1555e+03,  5.9750e+02],
                [ 4.5125e+01, -6.4625e+00,  1.1483e+03,  4.7073e+02],
                [ 4.1980e+01, -9.6911e-01,  5.3828e+02,  2.9217e+01],
                [ 3.7446e+01,  3.8211e+00,  1.1112e+03,  4.4523e+01],
                [ 4.9869e+01,  1.1388e+02,  4.6055e+02,  5.8804e+02],
                [ 3.8505e+01,  1.9076e+00,  1.1372e+03,  3.3271e+01],
                [ 9.5903e+00,  4.5630e-01,  1.0860e+03,  2.8386e+01],
                [ 5.0834e+02,  9.7218e+01,  1.1458e+03,  6.0320e+02],
                [ 3.8960e+01,  5.0640e+02,  1.1564e+03,  5.7847e+02],
                [ 1.6645e+02,  1.2937e+02,  9.9804e+02,  5.5823e+02],
                [ 4.5071e+01,  1.1615e+01,  1.1429e+03,  6.7996e+01],
                [ 4.1239e+01,  5.5591e+02,  1.1571e+03,  6.2312e+02],
                [ 4.3340e+01,  5.4890e+02,  1.1556e+03,  5.9665e+02],
                [ 4.9021e+01,  3.2059e+01,  1.1508e+03,  9.2955e+01],
                [ 6.9503e+01,  1.4246e+01,  7.7519e+02,  7.7584e+01],
                [ 4.1008e+01,  1.6785e+02,  1.1515e+03,  2.2429e+02],
                [ 1.2729e+02,  5.2241e+01,  8.1003e+02,  5.3105e+02],
                [ 4.6936e+01,  3.0524e+01,  1.1466e+03,  9.6479e+01],
                [ 9.5876e+01,  4.9414e+02,  1.0719e+03,  6.2183e+02],
                [ 4.2887e+01,  5.1232e+02,  1.1614e+03,  5.6759e+02],
                [ 4.4843e+01,  2.4753e+02,  1.1580e+03,  6.0352e+02],
                [ 7.0802e+01,  6.2028e+01,  8.5001e+02,  1.1084e+02],
                [ 5.3917e+01,  9.9795e+01,  5.1599e+02,  6.0565e+02],
                [ 3.2868e+01,  8.7996e+01,  1.1590e+03,  5.9229e+02],
                [ 4.5494e+01,  3.9683e+02,  1.1539e+03,  4.4549e+02],
                [ 6.6844e+01,  1.6838e+00,  1.0474e+03,  3.4740e+01],
                [ 4.5477e+01,  2.6104e+00,  1.1443e+03,  4.9333e+01],
                [ 8.6648e+01, -2.3576e-02,  1.0275e+03,  3.9996e+01],
                [ 4.9709e+01,  4.8222e+00,  1.1475e+03,  3.8126e+01],
                [ 9.6671e+01,  3.0668e+01,  1.0406e+03,  7.2303e+01],
                [ 5.6844e+01,  4.9612e+01,  6.0458e+02,  1.1539e+02],
                [ 6.3264e+01,  8.0691e+01,  4.6804e+02,  5.9280e+02],
                [ 7.6852e+01, -1.1565e+00,  7.4199e+02,  2.1250e+01],
                [ 6.9130e+01,  3.1369e+01,  1.1133e+03,  4.7971e+02],
                [ 4.5735e+01,  5.3470e+02,  1.1604e+03,  6.0881e+02],
                [ 5.1265e+01,  1.1047e+02,  4.6620e+02,  5.9231e+02],
                [ 9.9193e+01,  7.3642e+01,  6.6282e+02,  5.9093e+02],
                [ 4.8412e+01,  9.9693e+01,  5.2272e+02,  5.9783e+02],
                [ 4.5421e+01,  1.3995e+02,  1.1528e+03,  6.0731e+02],
                [ 4.2739e+01,  4.9547e+00,  1.1480e+03,  5.5840e+01],
                [ 4.1792e+01,  9.8066e+00,  1.1254e+03,  5.4586e+01],
                [ 4.5367e+01, -3.1910e+00,  3.8108e+02,  1.8262e+01],
                [ 4.2681e+01,  2.6223e+00,  1.1460e+03,  3.8930e+01],
                [ 5.2421e+01,  1.7896e+02,  1.1326e+03,  4.9651e+02],
                [ 3.8130e+01,  3.6169e+00,  1.1500e+03,  5.3515e+01],
                [ 4.5479e+01,  4.0711e+00,  1.1400e+03,  4.5629e+01],
                [ 4.5609e+01,  3.4607e+02,  1.1540e+03,  3.9446e+02],
                [ 7.2973e+01,  5.3067e+02,  7.2236e+02,  6.0417e+02],
                [ 9.4728e+01,  4.6616e+01,  7.9715e+02,  5.5978e+02],
                [ 5.1606e+01,  2.8581e+00,  1.1433e+03,  3.8877e+01],
                [ 3.6375e+01,  2.3434e+01,  1.1651e+03,  6.1006e+02],
                [ 9.2162e+01,  4.2754e+01,  5.9753e+02,  5.7560e+02],
                [ 4.8216e+01,  4.3356e+00,  1.1462e+03,  3.5780e+01]])
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(bbox1, bbox2):
    """
    Compute the Intersection over Box (IoB) ratio for `bbox1` with respect to `bbox2`.

    Parameters:
        bbox1 (tuple or list): The first bounding box in the format (x_min, y_min, x_max, y_max).
        bbox2 (tuple or list): The second bounding box in the same format (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoB ratio, which is the intersection area divided by the area of `bbox1`.
               Returns 0 if `bbox1` has zero area.

    Example:
        >>> bbox1 = (50, 50, 150, 150)
        >>> bbox2 = (100, 100, 200, 200)
        >>> iob(bbox1, bbox2)
        0.25

        >>> bbox1 = [30.00767707824707, 278.62286376953125, 1043.2724609375, 319.4166564941406]
        >>> bbox2 = [24.253061294555664, 23.11696434020996, 1045.835693359375, 333.2397766113281]
        >>> iob(bbox1, bbox2)
        1.0

    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        iob_ratio = intersection.get_area() / bbox1_area
        return iob_ratio

    return 0


def enhance_image(image):
    """This function enhances a given image by applying several preprocessing steps,
    including converting to grayscale, reducing noise with Gaussian blur, binarization,
    and sharpening. It returns the processed image as a PIL Image object.

    Parameters:
        image: a PIL Image object representing the input image to be enhanced.

    Returns:
        final_img: a PIL Image object representing the enhanced, sharpened, and binarized
                   version of the input image.

    Example:
        #>>> img = Image.open("sample_image.jpg")
        #>>> enhanced_img = enhance_image(img)
        #>>> enhanced_img.show()  # Opens the enhanced image for display
    """

    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 1. Convert to Grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Apply GaussianBlur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 3. Apply Thresholding (Binarization)
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Sharpen the Image using kernel filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(binary_img, -1, kernel)

    # 5. Rescale the image (if too small, you can upscale it)
    #resized_img = cv2.resize(sharpened_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Convert back to PIL Image for saving
    final_img = Image.fromarray(sharpened_img)

    return final_img

def align_headers(headers, rows):
    """
    Adjust the header boundary to form the convex hull of rows it intersects
    by at least 50% of the row's height.

    Note:
        This implementation assumes a single header per table and does not support
        multiple headers. It eliminates any headers beyond the top-most one.

    Parameters:
        headers (list of dict): List of header dictionaries, where each dictionary
                                contains at least a 'bbox' key representing the bounding box
                                of the header (format: [x_min, y_min, x_max, y_max]).
        rows (list of dict): List of row dictionaries, where each dictionary contains at least
                             a 'bbox' key representing the bounding box of the row
                             (format: [x_min, y_min, x_max, y_max]).

    Returns:
        list of dict: A list containing a single aligned header dictionary with an updated 'bbox'.

    Example:
        >>> headers = [{'bbox': [0.5, 0, 2.5, 2]}]  # Header overlaps the first two columns
        >>> rows = [
        >>>     {'bbox': [0, 0, 1, 10]},  # Column 1
        >>>     {'bbox': [1, 0, 3, 10]},  # Column 2
        >>>     {'bbox': [3, 0, 6, 10]},  # Column 3
        >>>     {'bbox': [6, 0, 9, 10]}   # Column 4
        >>> ]
        >>> align_headers(headers, rows)
        [{'bbox': [0, 0, 3, 2]}]

    """

    aligned_headers = []

    for row in rows:
        row['column header'] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row['column header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers

def align_headers_cols(headers, cols):
    """
    Adjust the header boundary to form the convex hull of columns it intersects
    by at least 50% of the column's width.

    Note:
        This implementation assumes a single header per table and does not support
        multiple headers. It eliminates any headers beyond the left-most one.

    Parameters:
        headers (list of dict): List of header dictionaries, where each dictionary
                                contains at least a 'bbox' key representing the bounding box
                                of the header (format: [x_min, y_min, x_max, y_max]).
        rows (list of dict): List of column dictionaries, where each dictionary contains at least
                             a 'bbox' key representing the bounding box of the column
                             (format: [x_min, y_min, x_max, y_max]).

    Returns:
        list of dict: A list containing a single aligned header dictionary with an updated 'bbox'.

    """

    aligned_headers = []

    for col in cols:
        col['row header'] = False

    header_col_nums = []
    for header in headers:
        for col_num, col in enumerate(cols):
            #row_height = row['bbox'][3] - row['bbox'][1]
            col_width = col['bbox'][4] - col['bbox'][2]
            min_col_overlap = max(col['bbox'][2], header['bbox'][2])
            max_col_overlap = min(col['bbox'][4], header['bbox'][4])
            overlap_width = max_col_overlap - min_col_overlap
            if overlap_width / col_width >= 0.5:
                header_col_nums.append(col_num)

    if len(header_col_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_col_nums[0] > 0:
        header_col_nums = list(range(header_col_nums[0] + 1)) + header_col_nums

    last_col_num = -1
    for col_num in header_col_nums:
        if col_num == last_col_num + 1:
            col = cols[col_num]
            col['row header'] = True
            header_rect = header_rect.include_rect(col['bbox'])
            last_col_num = col_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers

def refine_table_structure(table_structure, class_thresholds):
    """
    Refine the detected table structure by applying thresholding, NMS, and alignment operations.

    Parameters:
        table_structure (dict): Dictionary containing the detected table structure, including rows,
                                 columns, headers, spanning cells, etc.
        class_thresholds (dict): Dictionary containing the thresholds for different table elements
                                 like column headers, row headers, spanning cells, etc.

    Returns:
        dict: Refined table structure with processed rows, columns, headers, and spanning cells.

    Example:
        >>> table_structure = {'table name': {'bbox': [30.00767707824707, 278.62286376953125, 1043.2724609375, 319.4166564941406], 'position': 'above_row'}, 'rows': [{'label': 'table row', 'score': 0.9656466245651245, 'bbox': [37.289833068847656, 11.574613571166992, 1040.6044921875, 75.79998779296875], 'column header': False}, {'label': 'table row', 'score': 0.9897612929344177, 'bbox': [37.289833068847656, 77.70536041259766, 1040.6044921875, 123.46636962890625], 'column header': False}, {'label': 'table row', 'score': 0.9910268187522888, 'bbox': [37.289833068847656, 127.3360824584961, 1040.6044921875, 172.37257385253906], 'column header': False}, {'label': 'table row', 'score': 0.9891771674156189, 'bbox': [37.289833068847656, 176.07571411132812, 1040.6044921875, 220.73428344726562], 'column header': False}, {'label': 'table row', 'score': 0.9881728291511536, 'bbox': [37.289833068847656, 223.12820434570312, 1040.6044921875, 266.7409973144531], 'column header': False}, {'label': 'table row', 'score': 0.9894185662269592, 'bbox': [37.289833068847656, 269.58770751953125, 1040.6044921875, 315.38812255859375], 'column header': False}], 'columns': [{'label': 'table column', 'score': 0.9318616986274719, 'bbox': [37.289833068847656, 11.574613571166992, 440.29986572265625, 315.38812255859375], 'row header': True}, {'label': 'table column', 'score': 0.9548751711845398, 'bbox': [447.4510803222656, 11.574613571166992, 1040.6044921875, 315.38812255859375], 'row header': False}], 'column headers': [], 'row headers': [{'label': 'table row header', 'score': 0.8140369057655334, 'bbox': [33.017822265625, 81.90345001220703, 435.8532409667969, 322.3648376464844]}], 'spanning cells': [{'label': 'table spanning cell', 'score': 0.4390396773815155, 'bbox': [29.895198822021484, 11.369844436645508, 1041.8055419921875, 74.07972717285156], 'projected row header': False, 'projected column header': False}], 'both headers': False}
        >>> class_thresholds = {'no object': 10, 'table': 0.8, 'table column': 0.6, 'table column header': 0.6, 'table name': 0.6, 'table projected column header': 0.3, 'table projected row header': 0.3, 'table row': 0.6, 'table row header': 0.6, 'table spanning cell': 0.4}
        >>> refined_structure = refine_table_structure(table_structure, class_thresholds)
        >>> print(refined_structure['columns'])
        [{'label': 'table column', 'score': 0.9318616986274719, 'bbox': [37.289833068847656, 11.574613571166992, 440.29986572265625, 315.38812255859375], 'row header': True, 'column header': True}, {'label': 'table column', 'score': 0.9548751711845398, 'bbox': [447.4510803222656, 11.574613571166992, 1040.6044921875, 315.38812255859375], 'row header': False, 'column header': True}]
        >>> print(refined_structure['rows'])
        [{'label': 'table row', 'score': 0.9656466245651245, 'bbox': [37.289833068847656, 11.574613571166992, 1040.6044921875, 75.79998779296875], 'column header': False}, {'label': 'table row', 'score': 0.9897612929344177, 'bbox': [37.289833068847656, 77.70536041259766, 1040.6044921875, 123.46636962890625], 'column header': False}, {'label': 'table row', 'score': 0.9910268187522888, 'bbox': [37.289833068847656, 127.3360824584961, 1040.6044921875, 172.37257385253906], 'column header': False}, {'label': 'table row', 'score': 0.9891771674156189, 'bbox': [37.289833068847656, 176.07571411132812, 1040.6044921875, 220.73428344726562], 'column header': False}, {'label': 'table row', 'score': 0.9881728291511536, 'bbox': [37.289833068847656, 223.12820434570312, 1040.6044921875, 266.7409973144531], 'column header': False}, {'label': 'table row', 'score': 0.9894185662269592, 'bbox': [37.289833068847656, 269.58770751953125, 1040.6044921875, 315.38812255859375], 'column header': False}]

    """
    rows = table_structure["rows"]
    columns = table_structure['columns']
    table_name = table_structure['table name']

    # Process the headers
    column_headers = table_structure['column headers']
    column_headers = postprocess.apply_threshold(column_headers, class_thresholds["table column header"])
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process the row headers
    row_headers = table_structure['row headers']
    row_headers = postprocess.apply_threshold(row_headers, class_thresholds["table row header"])
    row_headers = postprocess.nms(row_headers)
    row_headers = align_headers(row_headers, columns)
    #row_headers = align_headers_cols(row_headers, columns)

    # Process spanning cells
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header'] and not elem['projected column header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]

    projected_column_headers = [elem for elem in table_structure['spanning cells'] if elem['projected column header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    projected_column_headers = postprocess.apply_threshold(projected_column_headers,
                                                        class_thresholds["table projected column header"])
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure['table_name'] = table_name
    table_structure['columns'] = columns
    table_structure['rows'] = rows
    table_structure['spanning cells'] = spanning_cells
    table_structure['column headers'] = column_headers
    table_structure['row headers'] = row_headers

    return table_structure

def extract_words_from_images(input_folder, output_folder):
    '''This function takes 2 paths as input and for all image files in the input_folder, it extracts words using OCR
        and saves the extracted data as JSON in the output_folder.

        Parameters:
            input_folder (str): Path to the folder containing images.
            output_folder (str): Path to the folder where JSON output will be saved.
    '''

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Identify all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', 'PNG', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        try:
            # Finding path to input images
            image_path = os.path.join(input_folder, image_file)
            with Image.open(image_path).convert('RGB') as img:
                # improve the input image
                img = img.filter(ImageFilter.SHARPEN)
                img = enhance_image(img)
                # Use Tesseract to do OCR on the image
                extracted_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                #spell = SpellChecker()

                words_data = [
                    {"bbox": [extracted_data['left'][i], extracted_data['top'][i],
                              extracted_data['left'][i] + extracted_data['width'][i],
                              extracted_data['top'][i] + extracted_data['height'][i]],
                     "text": extracted_data['text'][i].strip(),
                     "line_num": extracted_data['line_num'][i],
                     "block_num": extracted_data['block_num'][i]}
                    for i in range(len(extracted_data['text'])) if extracted_data['text'][i].strip()
                ]

                # Construct the full path to the output JSON file (same name as image file with .json extension)
                output_json_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + "_words.json")

                # Write the extracted word data to a JSON file
                with open(output_json_path, 'w') as json_file:
                    json.dump(words_data, json_file)
                    print("OCR file created")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def outputs_to_objects(outputs, img_size, class_idx2name):
    """
        Convert the outputs of an object detection model to a list of detected objects with their labels, scores, and bounding boxes.

        Parameters:
            outputs (dict): The outputs from the object detection model, typically containing:
                             - 'pred_logits': Logits for predicted class probabilities.
                             - 'pred_boxes': Predicted bounding boxes in the format [cx, cy, w, h].
            img_size (tuple): The size of the input image (width, height) to rescale the bounding boxes to image coordinates.
            class_idx2name (dict): A dictionary mapping class indices (integers) to class labels (strings).

        Returns:
            list: A list of dictionaries, each representing a detected object with the following keys:
                  - 'label': The class label of the object.
                  - 'score': The detection score of the object.
                  - 'bbox': The bounding box of the object in the format [x_min, y_min, x_max, y_max].

        Example:
            >>> outputs = {'pred_logits': tensor([[[ 2.2907e+00, -1.2149e+01, -3.6959e+00],
                     [ 1.5523e+00, -1.2637e+01, -2.5767e+00],
                     [-5.6445e+00, -1.9113e+01,  5.7241e+00],
                     [-4.1300e+00, -1.7870e+01,  1.1559e+00],
                     [-5.7814e+00, -1.7619e+01,  5.6715e+00],
                     [-5.2012e+00, -1.7749e+01,  2.9235e+00],
                     [ 3.8275e-01, -1.6138e+01,  2.5757e-02],
                     [ 1.1744e+00, -1.3235e+01, -3.6358e+00],
                     [-5.7238e+00, -1.7612e+01,  5.5349e+00],
                     [ 1.9355e+00, -1.4988e+01, -1.1543e+00],
                     [-4.0946e+00, -2.6952e+01,  1.9987e+00],
                     [-4.8150e+00, -1.5060e+01,  2.2194e+00],
                     [ 1.7070e-01, -1.4452e+01, -1.0239e+00],
                     [ 3.7653e+00, -7.6582e+00, -2.6189e+00],
                     [-5.0326e+00, -1.8416e+01,  3.0039e+00]]], grad_fn=<SelectBackward0>), 'pred_boxes': tensor([[[0.7144, 0.7041, 0.4671, 0.1790],
                     [0.7242, 0.8372, 0.4353, 0.0722],
                     [0.7185, 0.2427, 0.4608, 0.1866],
                     [0.2639, 0.8494, 0.4279, 0.1019],
                     [0.7231, 0.2556, 0.4637, 0.1933],
                     [0.7020, 0.8345, 0.4775, 0.0771],
                     [0.7228, 0.4020, 0.4662, 0.0838],
                     [0.2587, 0.8531, 0.4175, 0.1005],
                     [0.7288, 0.2515, 0.4436, 0.1952],
                     [0.7187, 0.4130, 0.4697, 0.0883],
                     [0.2520, 0.7219, 0.3947, 0.1001],
                     [0.7204, 0.6813, 0.4551, 0.2093],
                     [0.7178, 0.5298, 0.4666, 0.1485],
                     [0.7173, 0.2512, 0.4550, 0.2144],
                     [0.7182, 0.8302, 0.4473, 0.0782]]], grad_fn=<SelectBackward0>)}
            >>> img_size = (2469, 3378)
            >>> class_idx2name = {0: 'table', 1: 'table rotated', 2: 'no object'}
            >>> objects = outputs_to_objects(outputs, img_size, class_idx2name)
            >>> print(objects)
            [{'label': 'table', 'score': 0.9974936246871948, 'bbox': [1187.1407470703125, 2076.2255859375, 2340.4404296875, 2680.80712890625]}, {'label': 'table', 'score': 0.9841561913490295, 'bbox': [1250.6942138671875, 2706.241455078125, 2325.348388671875, 2950.024169921875]}, {'label': 'table', 'score': 0.5883128643035889, 'bbox': [1209.153076171875, 1216.511962890625, 2360.174072265625, 1499.4619140625]}, {'label': 'table', 'score': 0.9919195175170898, 'bbox': [123.34955596923828, 2711.88623046875, 1154.1181640625, 3051.33837890625]}, {'label': 'table', 'score': 0.9564689993858337, 'bbox': [1194.602294921875, 1246.01953125, 2354.17529296875, 1544.369140625]}, {'label': 'table', 'score': 0.7675632834434509, 'bbox': [1196.2545166015625, 1538.70654296875, 2348.283203125, 2040.3292236328125]}, {'label': 'table', 'score': 0.9983038902282715, 'bbox': [1209.3204345703125, 486.3968811035156, 2332.78759765625, 1210.5946044921875]}]

        Notes:
            - The bounding boxes are first rescaled based on the image size before being returned.
            - Only objects with a class label different from 'no object' are included in the results.
            - The bounding box format assumes [cx, cy, width, height], which is converted to [x_min, y_min, x_max, y_max].
        """

    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def objects_to_crops(img, tokens, objects, class_thresholds, padding=20):
    """This function extracts cropped image regions and their tokens)
    based on detected objects from the image, and adjusts the bounding boxes of
    tokens. Optionally adds padding to the crops and handles rotation
    for specific object labels (e.g., rotated tables).

    Parameters:
        img: the input image (PIL.Image) from which crops will be taken.
        tokens: a list of token dictionaries, each containing 'bbox' (bounding box
                coordinates) and other token information.
        objects: a list of dictionaries where each represents a detected object with
                 keys 'label', 'score', and 'bbox'.
        class_thresholds: a dictionary mapping object labels to the minimum confidence
                          score required for an object to be considered.
        padding: an integer specifying how many pixels of padding to add around each
                 object's bounding box (default is 20).

    Returns:
        table_crops: a list of dictionaries, each containing the cropped image ('image')
                     and the adjusted tokens ('tokens') found within the cropped region.

    Example:
        >>> img = Image.open("../images/Eging_2.jpg")
        >>> tokens = [{'bbox': [1791, 158, 1916, 247], 'text': '&', 'line_num': 1, 'block_num': 1, 'span_num': 0}, {'bbox': [1971, 193, 2305, 238], 'text': 'EcInGcpv', 'line_num': 1, 'block_num': 1, 'span_num': 1}, {'bbox': [1770, 259, 1948, 282], 'text': 'KEENSTAR', 'line_num': 1, 'block_num': 2, 'span_num': 2}, {'bbox': [165, 277, 1060, 360], 'text': 'EG-455M72-HE/BF-DG', 'line_num': 1, 'block_num': 3, 'span_num': 3}, {'bbox': [145, 495, 426, 586], 'text': 'pEngineering', 'line_num': 1, 'block_num': 5, 'span_num': 4}, {'bbox': [433, 495, 927, 586], 'text': 'Drawings', 'line_num': 1, 'block_num': 5, 'span_num': 5}, {'bbox': [1244, 495, 1447, 586], 'text': 'Electrical', 'line_num': 1, 'block_num': 5, 'span_num': 6}, {'bbox': [1472, 495, 1760, 586], 'text': 'Characteristics', 'line_num': 1, 'block_num': 5, 'span_num': 7}, {'bbox': [986, 587, 997, 591], 'text': '~', 'line_num': 1, 'block_num': 22, 'span_num': 8}, {'bbox': [1265, 575, 1335, 594], 'text': 'Power', 'line_num': 1, 'block_num': 22, 'span_num': 9}, {'bbox': [1343, 574, 1393, 594], 'text': 'level', 'line_num': 1, 'block_num': 22, 'span_num': 10}, {'bbox': [1733, 575, 1775, 594], 'text': '435', 'line_num': 1, 'block_num': 22, 'span_num': 11}, {'bbox': [1851, 570, 1897, 605], 'text': '440', 'line_num': 1, 'block_num': 22, 'span_num': 12}, {'bbox': [1973, 570, 2018, 605], 'text': '445', 'line_num': 1, 'block_num': 22, 'span_num': 13}, {'bbox': [2099, 575, 2140, 594], 'text': '450', 'line_num': 1, 'block_num': 22, 'span_num': 14}, {'bbox': [2221, 575, 2262, 594], 'text': '455', 'line_num': 1, 'block_num': 22, 'span_num': 15}, {'bbox': [603, 622, 1125, 632], 'text': 'a', 'line_num': 2, 'block_num': 22, 'span_num': 16}, {'bbox': [1265, 625, 1370, 648], 'text': 'Pmax(W)', 'line_num': 2, 'block_num': 22, 'span_num': 17}, {'bbox': [1733, 627, 1774, 646], 'text': '435', 'line_num': 2, 'block_num': 22, 'span_num': 18}, {'bbox': [1851, 618, 1896, 659], 'text': '440', 'line_num': 2, 'block_num': 22, 'span_num': 19}, {'bbox': [1976, 627, 2018, 646], 'text': '445', 'line_num': 2, 'block_num': 22, 'span_num': 20}, {'bbox': [2098, 627, 2140, 646], 'text': '450', 'line_num': 2, 'block_num': 22, 'span_num': 21}, {'bbox': [2220, 627, 2262, 646], 'text': '455', 'line_num': 2, 'block_num': 22, 'span_num': 22}, {'bbox': [1263, 670, 1307, 706], 'text': 'Vmp', 'line_num': 3, 'block_num': 22, 'span_num': 23}, {'bbox': [1316, 670, 1350, 706], 'text': '(Vv)', 'line_num': 3, 'block_num': 22, 'span_num': 24}, {'bbox': [1722, 674, 1785, 707], 'text': '41.04', 'line_num': 3, 'block_num': 22, 'span_num': 25}, {'bbox': [1844, 678, 1907, 697], 'text': '41.24', 'line_num': 3, 'block_num': 22, 'span_num': 26}, {'bbox': [1966, 678, 2029, 697], 'text': '41.44', 'line_num': 3, 'block_num': 22, 'span_num': 27}, {'bbox': [2088, 678, 2151, 697], 'text': '41.63', 'line_num': 3, 'block_num': 22, 'span_num': 28}, {'bbox': [2209, 678, 2272, 697], 'text': '41.82', 'line_num': 3, 'block_num': 22, 'span_num': 29}, {'bbox': [988, 725, 1079, 758], 'text': '—_..', 'line_num': 4, 'block_num': 22, 'span_num': 30}, {'bbox': [1265, 721, 1300, 757], 'text': 'Imp', 'line_num': 4, 'block_num': 22, 'span_num': 31}, {'bbox': [1312, 725, 1344, 752], 'text': '(A)', 'line_num': 4, 'block_num': 22, 'span_num': 32}, {'bbox': [1723, 728, 1784, 747], 'text': '10.60', 'line_num': 4, 'block_num': 22, 'span_num': 33}, {'bbox': [1845, 728, 1906, 747], 'text': '10.67', 'line_num': 4, 'block_num': 22, 'span_num': 34}, {'bbox': [1967, 728, 2028, 747], 'text': '10.74', 'line_num': 4, 'block_num': 22, 'span_num': 35}, {'bbox': [2089, 728, 2150, 747], 'text': '10.81', 'line_num': 4, 'block_num': 22, 'span_num': 36}, {'bbox': [2211, 728, 2272, 747], 'text': '10.88', 'line_num': 4, 'block_num': 22, 'span_num': 37}, {'bbox': [1263, 780, 1303, 799], 'text': 'Voc', 'line_num': 5, 'block_num': 22, 'span_num': 38}, {'bbox': [1310, 778, 1339, 801], 'text': '(V)', 'line_num': 5, 'block_num': 22, 'span_num': 39}, {'bbox': [1722, 781, 1785, 800], 'text': '49.25', 'line_num': 5, 'block_num': 22, 'span_num': 40}, {'bbox': [1844, 777, 1907, 810], 'text': '49.44', 'line_num': 5, 'block_num': 22, 'span_num': 41}, {'bbox': [1966, 781, 2029, 800], 'text': '49.65', 'line_num': 5, 'block_num': 22, 'span_num': 42}, {'bbox': [2088, 781, 2151, 800], 'text': '49.85', 'line_num': 5, 'block_num': 22, 'span_num': 43}, {'bbox': [2210, 781, 2272, 800], 'text': '$0.06', 'line_num': 5, 'block_num': 22, 'span_num': 44}, {'bbox': [1266, 831, 1294, 849], 'text': 'Isc', 'line_num': 6, 'block_num': 22, 'span_num': 45}, {'bbox': [1300, 828, 1330, 851], 'text': '(A)', 'line_num': 6, 'block_num': 22, 'span_num': 46}, {'bbox': [1723, 831, 1784, 849], 'text': '11.11', 'line_num': 6, 'block_num': 22, 'span_num': 47}, {'bbox': [1845, 830, 1906, 849], 'text': '11.17', 'line_num': 6, 'block_num': 22, 'span_num': 48}, {'bbox': [1967, 830, 2028, 849], 'text': '11.24', 'line_num': 6, 'block_num': 22, 'span_num': 49}, {'bbox': [2089, 830, 2150, 849], 'text': '11.31', 'line_num': 6, 'block_num': 22, 'span_num': 50}, {'bbox': [2211, 830, 2272, 849], 'text': '11.38', 'line_num': 6, 'block_num': 22, 'span_num': 51}, {'bbox': [1266, 878, 1347, 898], 'text': 'Module', 'line_num': 7, 'block_num': 22, 'span_num': 52}, {'bbox': [1355, 878, 1462, 904], 'text': 'efficiency', 'line_num': 7, 'block_num': 22, 'span_num': 53}, {'bbox': [1467, 877, 1503, 900], 'text': '(%)', 'line_num': 7, 'block_num': 22, 'span_num': 54}, {'bbox': [1723, 878, 1785, 897], 'text': '20.01', 'line_num': 7, 'block_num': 22, 'span_num': 55}, {'bbox': [1845, 879, 1907, 898], 'text': '20.24', 'line_num': 7, 'block_num': 22, 'span_num': 56}, {'bbox': [1966, 879, 2028, 898], 'text': '20.47', 'line_num': 7, 'block_num': 22, 'span_num': 57}, {'bbox': [2088, 878, 2150, 897], 'text': '20.70', 'line_num': 7, 'block_num': 22, 'span_num': 58}, {'bbox': [2210, 879, 2272, 898], 'text': '20.93', 'line_num': 7, 'block_num': 22, 'span_num': 59}, {'bbox': [723, 937, 824, 955], 'text': 'Junction', 'line_num': 8, 'block_num': 22, 'span_num': 60}, {'bbox': [836, 936, 874, 955], 'text': 'bos', 'line_num': 8, 'block_num': 22, 'span_num': 61}, {'bbox': [988, 942, 1054, 947], 'text': '—', 'line_num': 8, 'block_num': 22, 'span_num': 62}, {'bbox': [1265, 933, 1375, 953], 'text': 'Maximum', 'line_num': 8, 'block_num': 22, 'span_num': 63}, {'bbox': [1383, 935, 1460, 959], 'text': 'system', 'line_num': 8, 'block_num': 22, 'span_num': 64}, {'bbox': [1462, 927, 1543, 963], 'text': 'voltage', 'line_num': 8, 'block_num': 22, 'span_num': 65}, {'bbox': [1552, 927, 1585, 963], 'text': '(V)', 'line_num': 8, 'block_num': 22, 'span_num': 66}, {'bbox': [1972, 934, 2026, 953], 'text': '1500', 'line_num': 8, 'block_num': 22, 'span_num': 67}, {'bbox': [1265, 982, 1315, 1001], 'text': 'Fuse', 'line_num': 9, 'block_num': 22, 'span_num': 68}, {'bbox': [1324, 975, 1387, 1011], 'text': 'Rating', 'line_num': 9, 'block_num': 22, 'span_num': 69}, {'bbox': [1399, 975, 1429, 1011], 'text': '(A)', 'line_num': 9, 'block_num': 22, 'span_num': 70}, {'bbox': [1985, 982, 2011, 1001], 'text': '20', 'line_num': 9, 'block_num': 22, 'span_num': 71}, {'bbox': [759, 1013, 1016, 1044], 'text': '~~', 'line_num': 10, 'block_num': 22, 'span_num': 72}, {'bbox': [1061, 1017, 1115, 1050], 'text': 'oo', 'line_num': 10, 'block_num': 22, 'span_num': 73}, {'bbox': [1265, 1030, 1408, 1055], 'text': 'Temperature', 'line_num': 10, 'block_num': 22, 'span_num': 74}, {'bbox': [1415, 1029, 1532, 1049], 'text': 'coefficient', 'line_num': 10, 'block_num': 22, 'span_num': 75}, {'bbox': [1547, 1030, 1609, 1049], 'text': 'Pmax', 'line_num': 10, 'block_num': 22, 'span_num': 76}, {'bbox': [1615, 1028, 1677, 1051], 'text': '(%°C)', 'line_num': 10, 'block_num': 22, 'span_num': 77}, {'bbox': [1962, 1030, 2033, 1049], 'text': '-0.350', 'line_num': 10, 'block_num': 22, 'span_num': 78}, {'bbox': [731, 1064, 788, 1084], 'text': 'Back', 'line_num': 11, 'block_num': 22, 'span_num': 79}, {'bbox': [818, 1070, 829, 1083], 'text': 'sis', 'line_num': 11, 'block_num': 22, 'span_num': 80}, {'bbox': [988, 1094, 1054, 1098], 'text': '—', 'line_num': 11, 'block_num': 22, 'span_num': 81}, {'bbox': [1265, 1082, 1408, 1107], 'text': 'Temperature', 'line_num': 11, 'block_num': 22, 'span_num': 82}, {'bbox': [1415, 1081, 1532, 1101], 'text': 'coefficient', 'line_num': 11, 'block_num': 22, 'span_num': 83}, {'bbox': [1547, 1082, 1575, 1100], 'text': 'Isc', 'line_num': 11, 'block_num': 22, 'span_num': 84}, {'bbox': [1581, 1079, 1643, 1102], 'text': '(%°C)', 'line_num': 11, 'block_num': 22, 'span_num': 85}, {'bbox': [1974, 1081, 2021, 1101], 'text': '0.05', 'line_num': 11, 'block_num': 22, 'span_num': 86}, {'bbox': [1265, 1133, 1408, 1158], 'text': 'Temperature', 'line_num': 12, 'block_num': 22, 'span_num': 87}, {'bbox': [1415, 1132, 1532, 1152], 'text': 'coefficient', 'line_num': 12, 'block_num': 22, 'span_num': 88}, {'bbox': [1545, 1135, 1585, 1153], 'text': 'Voc', 'line_num': 12, 'block_num': 22, 'span_num': 89}, {'bbox': [1591, 1132, 1653, 1155], 'text': '(%°C)', 'line_num': 12, 'block_num': 22, 'span_num': 90}, {'bbox': [1962, 1135, 2033, 1154], 'text': '-0.275', 'line_num': 12, 'block_num': 22, 'span_num': 91}, {'bbox': [1266, 1181, 1436, 1201], 'text': '$TC:lrradiance', 'line_num': 13, 'block_num': 22, 'span_num': 92}, {'bbox': [1451, 1179, 1572, 1205], 'text': '1000W/m?,', 'line_num': 13, 'block_num': 22, 'span_num': 93}, {'bbox': [1580, 1181, 1664, 1201], 'text': 'module', 'line_num': 13, 'block_num': 22, 'span_num': 94}, {'bbox': [1671, 1184, 1819, 1207], 'text': 'temperature', 'line_num': 13, 'block_num': 22, 'span_num': 95}, {'bbox': [1829, 1182, 1887, 1203], 'text': '25°C,', 'line_num': 13, 'block_num': 22, 'span_num': 96}, {'bbox': [1904, 1183, 1977, 1201], 'text': 'AM=1.5', 'line_num': 13, 'block_num': 22, 'span_num': 97}, {'bbox': [1264, 1248, 1349, 1269], 'text': 'Bifacial', 'line_num': 14, 'block_num': 22, 'span_num': 98}, {'bbox': [1357, 1248, 1555, 1274], 'text': 'Output-Backside', 'line_num': 14, 'block_num': 22, 'span_num': 99}, {'bbox': [1564, 1249, 1637, 1268], 'text': 'Power', 'line_num': 14, 'block_num': 22, 'span_num': 100}, {'bbox': [1644, 1248, 1696, 1268], 'text': 'Gain', 'line_num': 14, 'block_num': 22, 'span_num': 101}, {'bbox': [982, 1287, 1079, 1294], 'text': '—', 'line_num': 15, 'block_num': 22, 'span_num': 102}, {'bbox': [1135, 1290, 1168, 1327], 'text': 'Ty', 'line_num': 15, 'block_num': 22, 'span_num': 103}, {'bbox': [1334, 1292, 1438, 1327], 'text': 'Pmax(W)', 'line_num': 15, 'block_num': 22, 'span_num': 104}, {'bbox': [1733, 1298, 1775, 1317], 'text': '478', 'line_num': 15, 'block_num': 22, 'span_num': 105}, {'bbox': [1854, 1294, 1896, 1327], 'text': '484', 'line_num': 15, 'block_num': 22, 'span_num': 106}, {'bbox': [1975, 1298, 2017, 1317], 'text': '489', 'line_num': 15, 'block_num': 22, 'span_num': 107}, {'bbox': [2100, 1298, 2141, 1317], 'text': '495', 'line_num': 15, 'block_num': 22, 'span_num': 108}, {'bbox': [2222, 1298, 2262, 1318], 'text': '500', 'line_num': 15, 'block_num': 22, 'span_num': 109}, {'bbox': [982, 1346, 1103, 1379], 'text': '————!', 'line_num': 16, 'block_num': 22, 'span_num': 110}, {'bbox': [1264, 1324, 1313, 1343], 'text': '10%', 'line_num': 16, 'block_num': 22, 'span_num': 111}, {'bbox': [1334, 1349, 1420, 1369], 'text': 'Module', 'line_num': 16, 'block_num': 22, 'span_num': 112}, {'bbox': [1428, 1349, 1540, 1375], 'text': 'efficiency', 'line_num': 16, 'block_num': 22, 'span_num': 113}, {'bbox': [1547, 1348, 1584, 1371], 'text': '(%)', 'line_num': 16, 'block_num': 22, 'span_num': 114}, {'bbox': [1724, 1350, 1786, 1369], 'text': '21.99', 'line_num': 16, 'block_num': 22, 'span_num': 115}, {'bbox': [1844, 1350, 1906, 1369], 'text': '22.27', 'line_num': 16, 'block_num': 22, 'span_num': 116}, {'bbox': [1966, 1350, 2028, 1369], 'text': '22.50', 'line_num': 16, 'block_num': 22, 'span_num': 117}, {'bbox': [2089, 1350, 2152, 1369], 'text': '22.77', 'line_num': 16, 'block_num': 22, 'span_num': 118}, {'bbox': [2211, 1350, 2273, 1369], 'text': '23.00', 'line_num': 16, 'block_num': 22, 'span_num': 119}, {'bbox': [603, 1387, 1125, 1419], 'text': 'ee', 'line_num': 17, 'block_num': 22, 'span_num': 120}, {'bbox': [1147, 1416, 1155, 1419], 'text': '*"eL', 'line_num': 17, 'block_num': 22, 'span_num': 121}, {'bbox': [1263, 1424, 1313, 1443], 'text': 's0%', 'line_num': 17, 'block_num': 22, 'span_num': 122}, {'bbox': [1334, 1398, 1438, 1424], 'text': 'Pmax(W)', 'line_num': 17, 'block_num': 22, 'span_num': 123}, {'bbox': [1735, 1399, 1775, 1418], 'text': '522', 'line_num': 17, 'block_num': 22, 'span_num': 124}, {'bbox': [1855, 1399, 1895, 1418], 'text': '528', 'line_num': 17, 'block_num': 22, 'span_num': 125}, {'bbox': [1976, 1399, 2017, 1418], 'text': '534', 'line_num': 17, 'block_num': 22, 'span_num': 126}, {'bbox': [2096, 1385, 2140, 1427], 'text': '540', 'line_num': 17, 'block_num': 22, 'span_num': 127}, {'bbox': [2222, 1399, 2262, 1418], 'text': '546', 'line_num': 17, 'block_num': 22, 'span_num': 128}, {'bbox': [282, 1543, 410, 1648], 'text': 'oP', 'line_num': 1, 'block_num': 24, 'span_num': 129}, {'bbox': [400, 1557, 404, 1565], 'text': ':', 'line_num': 1, 'block_num': 24, 'span_num': 130}, {'bbox': [603, 1592, 764, 1599], 'text': 'a', 'line_num': 1, 'block_num': 24, 'span_num': 131}, {'bbox': [905, 1557, 1017, 1596], 'text': 'oe', 'line_num': 1, 'block_num': 24, 'span_num': 132}, {'bbox': [1262, 1541, 1431, 1599], 'text': 'Working', 'line_num': 1, 'block_num': 24, 'span_num': 133}, {'bbox': [1538, 1541, 1743, 1596], 'text': 'acteristics', 'line_num': 1, 'block_num': 24, 'span_num': 134}, {'bbox': [631, 1603, 771, 1633], 'text': '|', 'line_num': 1, 'block_num': 34, 'span_num': 135}, {'bbox': [931, 1608, 1030, 1643], 'text': '—_-', 'line_num': 1, 'block_num': 34, 'span_num': 136}, {'bbox': [1267, 1613, 1340, 1632], 'text': 'Power', 'line_num': 1, 'block_num': 34, 'span_num': 137}, {'bbox': [1348, 1612, 1401, 1632], 'text': 'level', 'line_num': 1, 'block_num': 34, 'span_num': 138}, {'bbox': [1750, 1614, 1791, 1633], 'text': '435', 'line_num': 1, 'block_num': 34, 'span_num': 139}, {'bbox': [1867, 1614, 1909, 1633], 'text': '440', 'line_num': 1, 'block_num': 34, 'span_num': 140}, {'bbox': [1982, 1609, 2028, 1644], 'text': '445', 'line_num': 1, 'block_num': 34, 'span_num': 141}, {'bbox': [2105, 1614, 2147, 1633], 'text': '450', 'line_num': 1, 'block_num': 34, 'span_num': 142}, {'bbox': [2223, 1614, 2264, 1633], 'text': '455', 'line_num': 1, 'block_num': 34, 'span_num': 143}, {'bbox': [1267, 1666, 1331, 1685], 'text': 'Pmax', 'line_num': 1, 'block_num': 34, 'span_num': 144}, {'bbox': [1337, 1664, 1374, 1687], 'text': '(W)', 'line_num': 1, 'block_num': 34, 'span_num': 145}, {'bbox': [1751, 1666, 1791, 1685], 'text': '323', 'line_num': 1, 'block_num': 34, 'span_num': 146}, {'bbox': [1868, 1666, 1909, 1685], 'text': '327', 'line_num': 1, 'block_num': 34, 'span_num': 147}, {'bbox': [1987, 1666, 2027, 1685], 'text': '330', 'line_num': 1, 'block_num': 34, 'span_num': 148}, {'bbox': [2106, 1666, 2146, 1685], 'text': '334', 'line_num': 1, 'block_num': 34, 'span_num': 149}, {'bbox': [2224, 1666, 2264, 1685], 'text': '337', 'line_num': 1, 'block_num': 34, 'span_num': 150}, {'bbox': [1160, 1704, 1161, 1762], 'text': '|', 'line_num': 1, 'block_num': 34, 'span_num': 151}, {'bbox': [1265, 1717, 1317, 1742], 'text': 'Vp', 'line_num': 1, 'block_num': 34, 'span_num': 152}, {'bbox': [1324, 1715, 1352, 1738], 'text': '(Vv)', 'line_num': 1, 'block_num': 34, 'span_num': 153}, {'bbox': [1740, 1718, 1802, 1737], 'text': '37.83', 'line_num': 1, 'block_num': 34, 'span_num': 154}, {'bbox': [1857, 1718, 1919, 1737], 'text': '38.03', 'line_num': 1, 'block_num': 34, 'span_num': 155}, {'bbox': [1976, 1718, 2038, 1737], 'text': '38.11', 'line_num': 1, 'block_num': 34, 'span_num': 156}, {'bbox': [2095, 1718, 2157, 1737], 'text': '38.35', 'line_num': 1, 'block_num': 34, 'span_num': 157}, {'bbox': [2213, 1718, 2275, 1737], 'text': '38.43', 'line_num': 1, 'block_num': 34, 'span_num': 158}, {'bbox': [1267, 1760, 1304, 1795], 'text': 'Imp', 'line_num': 1, 'block_num': 34, 'span_num': 159}, {'bbox': [1315, 1764, 1347, 1791], 'text': '(A)', 'line_num': 1, 'block_num': 34, 'span_num': 160}, {'bbox': [1747, 1763, 1795, 1796], 'text': '8.54', 'line_num': 1, 'block_num': 34, 'span_num': 161}, {'bbox': [1864, 1767, 1912, 1786], 'text': '8.60', 'line_num': 1, 'block_num': 34, 'span_num': 162}, {'bbox': [1983, 1767, 2031, 1786], 'text': '8.66', 'line_num': 1, 'block_num': 34, 'span_num': 163}, {'bbox': [2102, 1767, 2150, 1786], 'text': '8.71', 'line_num': 1, 'block_num': 34, 'span_num': 164}, {'bbox': [2220, 1767, 2267, 1786], 'text': '8.77', 'line_num': 1, 'block_num': 34, 'span_num': 165}, {'bbox': [542, 1819, 613, 1835], 'text': 'LOGCIAL', 'line_num': 1, 'block_num': 34, 'span_num': 166}, {'bbox': [628, 1819, 656, 1836], 'text': 'Ti', 'line_num': 1, 'block_num': 34, 'span_num': 167}, {'bbox': [1265, 1818, 1306, 1837], 'text': 'Voc', 'line_num': 1, 'block_num': 34, 'span_num': 168}, {'bbox': [1313, 1816, 1342, 1839], 'text': '(V)', 'line_num': 1, 'block_num': 34, 'span_num': 169}, {'bbox': [1739, 1818, 1802, 1837], 'text': '45.60', 'line_num': 1, 'block_num': 34, 'span_num': 170}, {'bbox': [1856, 1818, 1919, 1837], 'text': '45.81', 'line_num': 1, 'block_num': 34, 'span_num': 171}, {'bbox': [1975, 1818, 2038, 1838], 'text': '45.99', 'line_num': 1, 'block_num': 34, 'span_num': 172}, {'bbox': [2094, 1818, 2157, 1837], 'text': '46.19', 'line_num': 1, 'block_num': 34, 'span_num': 173}, {'bbox': [2212, 1818, 2275, 1837], 'text': '46.01', 'line_num': 1, 'block_num': 34, 'span_num': 174}, {'bbox': [373, 1874, 400, 1908], 'text': 'z', 'line_num': 1, 'block_num': 34, 'span_num': 175}, {'bbox': [1267, 1869, 1296, 1887], 'text': 'Isc', 'line_num': 1, 'block_num': 34, 'span_num': 176}, {'bbox': [1303, 1866, 1332, 1889], 'text': '(A)', 'line_num': 1, 'block_num': 34, 'span_num': 177}, {'bbox': [1747, 1868, 1795, 1887], 'text': '8.96', 'line_num': 1, 'block_num': 34, 'span_num': 178}, {'bbox': [1864, 1868, 1912, 1887], 'text': '9.05', 'line_num': 1, 'block_num': 34, 'span_num': 179}, {'bbox': [1983, 1868, 2031, 1887], 'text': '9.10', 'line_num': 1, 'block_num': 34, 'span_num': 180}, {'bbox': [2102, 1868, 2150, 1887], 'text': '9.16', 'line_num': 1, 'block_num': 34, 'span_num': 181}, {'bbox': [2220, 1868, 2268, 1887], 'text': '9.20', 'line_num': 1, 'block_num': 34, 'span_num': 182}, {'bbox': [377, 1915, 397, 1957], 'text': 'e', 'line_num': 2, 'block_num': 34, 'span_num': 183}, {'bbox': [1267, 1919, 1340, 1938], 'text': 'Power', 'line_num': 2, 'block_num': 34, 'span_num': 184}, {'bbox': [1347, 1918, 1457, 1938], 'text': 'tolerance', 'line_num': 2, 'block_num': 34, 'span_num': 185}, {'bbox': [1463, 1917, 1500, 1940], 'text': '(%)', 'line_num': 2, 'block_num': 34, 'span_num': 186}, {'bbox': [1980, 1919, 2035, 1938], 'text': '0~+3', 'line_num': 2, 'block_num': 34, 'span_num': 187}, {'bbox': [376, 1959, 397, 2016], 'text': '5', 'line_num': 3, 'block_num': 34, 'span_num': 188}, {'bbox': [1267, 1969, 1331, 1988], 'text': 'NOCT', 'line_num': 3, 'block_num': 34, 'span_num': 189}, {'bbox': [1338, 1967, 1377, 1990], 'text': '(°C)', 'line_num': 3, 'block_num': 34, 'span_num': 190}, {'bbox': [1974, 1971, 2040, 1991], 'text': '4442', 'line_num': 3, 'block_num': 34, 'span_num': 191}, {'bbox': [1265, 2014, 1595, 2034], 'text': 'NOCT:Conditions:Irradiance', 'line_num': 1, 'block_num': 35, 'span_num': 192}, {'bbox': [1609, 2012, 1705, 2038], 'text': '800W/m?', 'line_num': 1, 'block_num': 35, 'span_num': 193}, {'bbox': [1720, 2014, 1816, 2038], 'text': 'ambient', 'line_num': 1, 'block_num': 35, 'span_num': 194}, {'bbox': [1824, 2017, 1972, 2040], 'text': 'temperature', 'line_num': 1, 'block_num': 35, 'span_num': 195}, {'bbox': [1981, 2015, 2040, 2036], 'text': '20°C,', 'line_num': 1, 'block_num': 35, 'span_num': 196}, {'bbox': [2053, 2014, 2101, 2034], 'text': 'wind', 'line_num': 1, 'block_num': 35, 'span_num': 197}, {'bbox': [2112, 2014, 2180, 2040], 'text': 'speed', 'line_num': 1, 'block_num': 35, 'span_num': 198}, {'bbox': [2189, 2014, 2246, 2038], 'text': '1m/s', 'line_num': 1, 'block_num': 35, 'span_num': 199}, {'bbox': [1260, 2111, 1486, 2145], 'text': 'Mechanical', 'line_num': 1, 'block_num': 36, 'span_num': 200}, {'bbox': [1500, 2111, 1799, 2145], 'text': 'Characteristics', 'line_num': 1, 'block_num': 36, 'span_num': 201}, {'bbox': [2318, 2099, 2319, 2157], 'text': '|', 'line_num': 1, 'block_num': 36, 'span_num': 202}, {'bbox': [1260, 2174, 1356, 2194], 'text': 'Number', 'line_num': 1, 'block_num': 48, 'span_num': 203}, {'bbox': [1362, 2174, 1386, 2194], 'text': 'of', 'line_num': 1, 'block_num': 48, 'span_num': 204}, {'bbox': [1392, 2174, 1443, 2194], 'text': 'cells', 'line_num': 1, 'block_num': 48, 'span_num': 205}, {'bbox': [1749, 2175, 1829, 2200], 'text': '144pcs', 'line_num': 1, 'block_num': 48, 'span_num': 206}, {'bbox': [1259, 2224, 1306, 2244], 'text': 'Size', 'line_num': 2, 'block_num': 48, 'span_num': 207}, {'bbox': [1313, 2224, 1337, 2244], 'text': 'of', 'line_num': 2, 'block_num': 48, 'span_num': 208}, {'bbox': [1343, 2224, 1382, 2244], 'text': 'cell', 'line_num': 2, 'block_num': 48, 'span_num': 209}, {'bbox': [1389, 2223, 1450, 2246], 'text': '(mm)', 'line_num': 2, 'block_num': 48, 'span_num': 210}, {'bbox': [1749, 2224, 1829, 2245], 'text': '166*83', 'line_num': 2, 'block_num': 48, 'span_num': 211}, {'bbox': [1259, 2277, 1315, 2302], 'text': 'Type', 'line_num': 3, 'block_num': 48, 'span_num': 212}, {'bbox': [1323, 2276, 1346, 2296], 'text': 'of', 'line_num': 3, 'block_num': 48, 'span_num': 213}, {'bbox': [1352, 2276, 1391, 2296], 'text': 'cell', 'line_num': 3, 'block_num': 48, 'span_num': 214}, {'bbox': [1749, 2276, 1812, 2295], 'text': 'Mono', 'line_num': 3, 'block_num': 48, 'span_num': 215}, {'bbox': [374, 2328, 400, 2353], 'text': 'e', 'line_num': 4, 'block_num': 48, 'span_num': 216}, {'bbox': [1255, 2324, 1372, 2344], 'text': 'Thickness', 'line_num': 4, 'block_num': 48, 'span_num': 217}, {'bbox': [1380, 2324, 1403, 2344], 'text': 'of', 'line_num': 4, 'block_num': 48, 'span_num': 218}, {'bbox': [1409, 2324, 1467, 2350], 'text': 'glass', 'line_num': 4, 'block_num': 48, 'span_num': 219}, {'bbox': [1474, 2323, 1535, 2346], 'text': '(mm)', 'line_num': 4, 'block_num': 48, 'span_num': 220}, {'bbox': [1749, 2324, 1783, 2343], 'text': '2.0', 'line_num': 4, 'block_num': 48, 'span_num': 221}, {'bbox': [374, 2350, 402, 2425], 'text': 'Fs', 'line_num': 5, 'block_num': 48, 'span_num': 222}, {'bbox': [1262, 2378, 1317, 2403], 'text': 'Type', 'line_num': 5, 'block_num': 48, 'span_num': 223}, {'bbox': [1325, 2377, 1349, 2397], 'text': 'of', 'line_num': 5, 'block_num': 48, 'span_num': 224}, {'bbox': [1355, 2377, 1423, 2397], 'text': 'frame', 'line_num': 5, 'block_num': 48, 'span_num': 225}, {'bbox': [1746, 2376, 1856, 2396], 'text': 'Anodized', 'line_num': 5, 'block_num': 48, 'span_num': 226}, {'bbox': [1864, 2376, 1984, 2396], 'text': 'aluminum', 'line_num': 5, 'block_num': 48, 'span_num': 227}, {'bbox': [1992, 2376, 2049, 2402], 'text': 'alloy', 'line_num': 5, 'block_num': 48, 'span_num': 228}, {'bbox': [382, 2424, 398, 2451], 'text': '5', 'line_num': 6, 'block_num': 48, 'span_num': 229}, {'bbox': [1262, 2426, 1365, 2446], 'text': 'Junction', 'line_num': 6, 'block_num': 48, 'span_num': 230}, {'bbox': [1374, 2426, 1414, 2446], 'text': 'box', 'line_num': 6, 'block_num': 48, 'span_num': 231}, {'bbox': [1749, 2426, 1798, 2445], 'text': 'IP68', 'line_num': 6, 'block_num': 48, 'span_num': 232}, {'bbox': [377, 2454, 398, 2470], 'text': '~', 'line_num': 7, 'block_num': 48, 'span_num': 233}, {'bbox': [1262, 2478, 1308, 2498], 'text': 'Size', 'line_num': 7, 'block_num': 48, 'span_num': 234}, {'bbox': [1315, 2478, 1339, 2498], 'text': 'of', 'line_num': 7, 'block_num': 48, 'span_num': 235}, {'bbox': [1356, 2472, 1434, 2509], 'text': 'module', 'line_num': 7, 'block_num': 48, 'span_num': 236}, {'bbox': [1443, 2477, 1502, 2500], 'text': '(mm)', 'line_num': 7, 'block_num': 48, 'span_num': 237}, {'bbox': [1748, 2478, 1912, 2498], 'text': '2094*1038*30', 'line_num': 7, 'block_num': 48, 'span_num': 238}, {'bbox': [1261, 2524, 1332, 2560], 'text': 'Weight', 'line_num': 8, 'block_num': 48, 'span_num': 239}, {'bbox': [1347, 2524, 1392, 2560], 'text': '(kg)', 'line_num': 8, 'block_num': 48, 'span_num': 240}, {'bbox': [1749, 2531, 1797, 2550], 'text': '27.5', 'line_num': 8, 'block_num': 48, 'span_num': 241}, {'bbox': [1259, 2573, 1480, 2607], 'text': 'Cables/connectors', 'line_num': 9, 'block_num': 48, 'span_num': 242}, {'bbox': [1748, 2577, 1882, 2602], 'text': '4mm?7,MC4', 'line_num': 9, 'block_num': 48, 'span_num': 243}, {'bbox': [1890, 2579, 2022, 2605], 'text': 'compatible', 'line_num': 9, 'block_num': 48, 'span_num': 244}, {'bbox': [612, 2618, 712, 2648], 'text': 'Voltage', 'line_num': 10, 'block_num': 48, 'span_num': 245}, {'bbox': [719, 2617, 752, 2643], 'text': '(V)', 'line_num': 10, 'block_num': 48, 'span_num': 246}, {'bbox': [1260, 2629, 1340, 2655], 'text': 'Length', 'line_num': 10, 'block_num': 48, 'span_num': 247}, {'bbox': [1348, 2629, 1371, 2649], 'text': 'of', 'line_num': 10, 'block_num': 48, 'span_num': 248}, {'bbox': [1378, 2629, 1443, 2649], 'text': 'Cabel', 'line_num': 10, 'block_num': 48, 'span_num': 249}, {'bbox': [1749, 2621, 1847, 2656], 'text': 'Portrait:', 'line_num': 10, 'block_num': 48, 'span_num': 250}, {'bbox': [1855, 2628, 2062, 2652], 'text': '+300mm/-300mm', 'line_num': 10, 'block_num': 48, 'span_num': 251}, {'bbox': [145, 2713, 167, 2801], 'text': '|', 'line_num': 11, 'block_num': 48, 'span_num': 252}, {'bbox': [181, 2713, 336, 2801], 'text': 'Packing', 'line_num': 11, 'block_num': 48, 'span_num': 253}, {'bbox': [149, 2717, 1161, 2779], 'text': 'Configuration', 'line_num': 11, 'block_num': 48, 'span_num': 254}, {'bbox': [177, 2793, 250, 2813], 'text': 'Pieces', 'line_num': 12, 'block_num': 48, 'span_num': 255}, {'bbox': [259, 2799, 297, 2819], 'text': 'per', 'line_num': 12, 'block_num': 48, 'span_num': 256}, {'bbox': [304, 2793, 370, 2819], 'text': 'pallet', 'line_num': 12, 'block_num': 48, 'span_num': 257}, {'bbox': [649, 2797, 676, 2816], 'text': '36', 'line_num': 12, 'block_num': 48, 'span_num': 258}, {'bbox': [1258, 2794, 1376, 2820], 'text': 'Operating', 'line_num': 12, 'block_num': 48, 'span_num': 259}, {'bbox': [1383, 2794, 1577, 2820], 'text': 'Temperature(°C)', 'line_num': 12, 'block_num': 48, 'span_num': 260}, {'bbox': [1749, 2790, 1827, 2809], 'text': '-40~85', 'line_num': 12, 'block_num': 48, 'span_num': 261}, {'bbox': [176, 2844, 223, 2864], 'text': 'Size', 'line_num': 13, 'block_num': 48, 'span_num': 262}, {'bbox': [230, 2844, 254, 2864], 'text': 'of', 'line_num': 13, 'block_num': 48, 'span_num': 263}, {'bbox': [261, 2844, 353, 2870], 'text': 'packing', 'line_num': 13, 'block_num': 48, 'span_num': 264}, {'bbox': [364, 2843, 425, 2866], 'text': '(mm)', 'line_num': 13, 'block_num': 48, 'span_num': 265}, {'bbox': [648, 2842, 841, 2862], 'text': '2130*1140*1190', 'line_num': 13, 'block_num': 48, 'span_num': 266}, {'bbox': [1258, 2844, 1376, 2870], 'text': 'Operating', 'line_num': 13, 'block_num': 48, 'span_num': 267}, {'bbox': [1384, 2844, 1531, 2870], 'text': 'humidity(°C)', 'line_num': 13, 'block_num': 48, 'span_num': 268}, {'bbox': [1749, 2846, 1804, 2865], 'text': '5~85', 'line_num': 13, 'block_num': 48, 'span_num': 269}, {'bbox': [176, 2894, 258, 2922], 'text': 'Weight', 'line_num': 14, 'block_num': 48, 'span_num': 270}, {'bbox': [265, 2894, 289, 2914], 'text': 'of', 'line_num': 14, 'block_num': 48, 'span_num': 271}, {'bbox': [296, 2894, 388, 2922], 'text': 'packing', 'line_num': 14, 'block_num': 48, 'span_num': 272}, {'bbox': [399, 2893, 442, 2922], 'text': '(kg)', 'line_num': 14, 'block_num': 48, 'span_num': 273}, {'bbox': [650, 2890, 704, 2909], 'text': '1040', 'line_num': 14, 'block_num': 48, 'span_num': 274}, {'bbox': [1257, 2894, 1374, 2914], 'text': 'Allowable', 'line_num': 14, 'block_num': 48, 'span_num': 275}, {'bbox': [1383, 2894, 1427, 2914], 'text': 'Hail', 'line_num': 14, 'block_num': 48, 'span_num': 276}, {'bbox': [1436, 2894, 1491, 2914], 'text': 'Load', 'line_num': 14, 'block_num': 48, 'span_num': 277}, {'bbox': [1749, 2897, 1822, 2915], 'text': '25mm', 'line_num': 14, 'block_num': 48, 'span_num': 278}, {'bbox': [1831, 2895, 1914, 2915], 'text': 'ice-ball', 'line_num': 14, 'block_num': 48, 'span_num': 279}, {'bbox': [1922, 2895, 1973, 2915], 'text': 'with', 'line_num': 14, 'block_num': 48, 'span_num': 280}, {'bbox': [1981, 2895, 2074, 2921], 'text': 'velocity', 'line_num': 14, 'block_num': 48, 'span_num': 281}, {'bbox': [2081, 2895, 2104, 2915], 'text': 'of', 'line_num': 14, 'block_num': 48, 'span_num': 282}, {'bbox': [2111, 2895, 2182, 2919], 'text': '23m/s', 'line_num': 14, 'block_num': 48, 'span_num': 283}, {'bbox': [177, 2943, 196, 2963], 'text': 'Pi', 'line_num': 15, 'block_num': 48, 'span_num': 284}, {'bbox': [289, 2943, 416, 2963], 'text': 'rcontainer', 'line_num': 15, 'block_num': 48, 'span_num': 285}, {'bbox': [650, 2939, 691, 2958], 'text': '792', 'line_num': 15, 'block_num': 48, 'span_num': 286}, {'bbox': [176, 2992, 223, 3012], 'text': 'Size', 'line_num': 16, 'block_num': 48, 'span_num': 287}, {'bbox': [230, 2992, 254, 3012], 'text': 'of', 'line_num': 16, 'block_num': 48, 'span_num': 288}, {'bbox': [260, 2992, 373, 3012], 'text': 'container', 'line_num': 16, 'block_num': 48, 'span_num': 289}, {'bbox': [647, 2984, 717, 3004], 'text': '40°HC', 'line_num': 16, 'block_num': 48, 'span_num': 290}, {'bbox': [1809, 3066, 1851, 3086], 'text': 'Tel:', 'line_num': 1, 'block_num': 50, 'span_num': 291}, {'bbox': [1857, 3067, 2057, 3086], 'text': '86-519-82585880', 'line_num': 1, 'block_num': 50, 'span_num': 292}, {'bbox': [1809, 3103, 1854, 3129], 'text': 'Zip:', 'line_num': 1, 'block_num': 50, 'span_num': 293}, {'bbox': [1862, 3104, 1943, 3123], 'text': '213213', 'line_num': 1, 'block_num': 50, 'span_num': 294}, {'bbox': [1808, 3140, 1863, 3160], 'text': 'Add:', 'line_num': 1, 'block_num': 50, 'span_num': 295}, {'bbox': [1870, 3136, 1942, 3169], 'text': 'No.18,', 'line_num': 1, 'block_num': 50, 'span_num': 296}, {'bbox': [1949, 3140, 2013, 3160], 'text': 'Jinwu', 'line_num': 1, 'block_num': 50, 'span_num': 297}, {'bbox': [2025, 3136, 2083, 3169], 'text': 'Road,', 'line_num': 1, 'block_num': 50, 'span_num': 298}, {'bbox': [2095, 3140, 2162, 3165], 'text': 'Jintan', 'line_num': 1, 'block_num': 50, 'span_num': 299}, {'bbox': [2175, 3140, 2225, 3165], 'text': 'Dist,', 'line_num': 1, 'block_num': 50, 'span_num': 300}, {'bbox': [1809, 3177, 1941, 3203], 'text': 'Changzhou', 'line_num': 2, 'block_num': 50, 'span_num': 301}, {'bbox': [1942, 3173, 2001, 3208], 'text': 'City,', 'line_num': 2, 'block_num': 50, 'span_num': 302}, {'bbox': [2006, 3177, 2091, 3203], 'text': 'Jiangsu', 'line_num': 2, 'block_num': 50, 'span_num': 303}, {'bbox': [2103, 3177, 2211, 3197], 'text': 'Province.', 'line_num': 2, 'block_num': 50, 'span_num': 304}, {'bbox': [1810, 3210, 1884, 3244], 'text': 'Email:', 'line_num': 3, 'block_num': 50, 'span_num': 305}, {'bbox': [1890, 3214, 2187, 3240], 'text': 'marketing@egingpv.com', 'line_num': 3, 'block_num': 50, 'span_num': 306}, {'bbox': [1809, 3251, 1868, 3271], 'text': 'Web:', 'line_num': 4, 'block_num': 50, 'span_num': 307}, {'bbox': [1874, 3251, 2091, 3277], 'text': 'www.egingpv.com', 'line_num': 4, 'block_num': 50, 'span_num': 308}, {'bbox': [152, 3173, 248, 3194], 'text': 'Revised', 'line_num': 1, 'block_num': 51, 'span_num': 309}, {'bbox': [256, 3173, 278, 3194], 'text': 'in', 'line_num': 1, 'block_num': 51, 'span_num': 310}, {'bbox': [287, 3175, 338, 3194], 'text': 'MAY', 'line_num': 1, 'block_num': 51, 'span_num': 311}, {'bbox': [345, 3175, 404, 3194], 'text': '2022', 'line_num': 1, 'block_num': 51, 'span_num': 312}, {'bbox': [413, 3174, 452, 3194], 'text': '1th', 'line_num': 1, 'block_num': 51, 'span_num': 313}, {'bbox': [461, 3173, 549, 3194], 'text': 'Edition', 'line_num': 1, 'block_num': 51, 'span_num': 314}, {'bbox': [150, 3214, 296, 3234], 'text': 'CAUTION:All', 'line_num': 1, 'block_num': 52, 'span_num': 315}, {'bbox': [305, 3214, 371, 3240], 'text': 'rights', 'line_num': 1, 'block_num': 52, 'span_num': 316}, {'bbox': [379, 3214, 480, 3234], 'text': 'reserved', 'line_num': 1, 'block_num': 52, 'span_num': 317}, {'bbox': [490, 3214, 516, 3240], 'text': 'by', 'line_num': 1, 'block_num': 52, 'span_num': 318}, {'bbox': [524, 3215, 597, 3235], 'text': 'EGING', 'line_num': 1, 'block_num': 52, 'span_num': 319}, {'bbox': [606, 3215, 639, 3234], 'text': 'PV.', 'line_num': 1, 'block_num': 52, 'span_num': 320}, {'bbox': [150, 3249, 317, 3275], 'text': 'Specifications', 'line_num': 2, 'block_num': 52, 'span_num': 321}, {'bbox': [325, 3249, 426, 3269], 'text': 'included', 'line_num': 2, 'block_num': 52, 'span_num': 322}, {'bbox': [435, 3249, 454, 3269], 'text': 'in', 'line_num': 2, 'block_num': 52, 'span_num': 323}, {'bbox': [463, 3249, 505, 3269], 'text': 'this', 'line_num': 2, 'block_num': 52, 'span_num': 324}, {'bbox': [513, 3249, 630, 3269], 'text': 'datasheet', 'line_num': 2, 'block_num': 52, 'span_num': 325}, {'bbox': [637, 3255, 673, 3269], 'text': 'are', 'line_num': 2, 'block_num': 52, 'span_num': 326}, {'bbox': [681, 3249, 766, 3275], 'text': 'subject', 'line_num': 2, 'block_num': 52, 'span_num': 327}, {'bbox': [774, 3251, 796, 3269], 'text': 'to', 'line_num': 2, 'block_num': 52, 'span_num': 328}, {'bbox': [804, 3249, 889, 3275], 'text': 'change', 'line_num': 2, 'block_num': 52, 'span_num': 329}, {'bbox': [897, 3249, 989, 3269], 'text': 'without', 'line_num': 2, 'block_num': 52, 'span_num': 330}, {'bbox': [997, 3249, 1075, 3269], 'text': 'notice.', 'line_num': 2, 'block_num': 52, 'span_num': 331}]
        >>> objects = [{'label': 'table', 'score': 0.9974936246871948, 'bbox': [1187.1407470703125, 2076.2255859375, 2340.4404296875, 2680.80712890625]}, {'label': 'table', 'score': 0.9841561913490295, 'bbox': [1250.6942138671875, 2706.241455078125, 2325.348388671875, 2950.024169921875]}, {'label': 'table', 'score': 0.5883128643035889, 'bbox': [1209.153076171875, 1216.511962890625, 2360.174072265625, 1499.4619140625]}, {'label': 'table', 'score': 0.9919195175170898, 'bbox': [123.34955596923828, 2711.88623046875, 1154.1181640625, 3051.33837890625]}, {'label': 'table', 'score': 0.9564689993858337, 'bbox': [1194.602294921875, 1246.01953125, 2354.17529296875, 1544.369140625]}, {'label': 'table', 'score': 0.7675632834434509, 'bbox': [1196.2545166015625, 1538.70654296875, 2348.283203125, 2040.3292236328125]}, {'label': 'table', 'score': 0.9983038902282715, 'bbox': [1209.3204345703125, 486.3968811035156, 2332.78759765625, 1210.5946044921875]}]
        >>> class_thresholds = {'no object': 10, 'table': 0.8, 'table rotated': 0.7}
        >>> crops = objects_to_crops(img, tokens, objects, class_thresholds)
        >>> crops[0]['tokens']
        [{'bbox': [92.8592529296875, 54.7744140625, 318.8592529296875, 88.7744140625], 'text': 'Mechanical', 'line_num': 1, 'block_num': 36, 'span_num': 200}, {'bbox': [332.8592529296875, 54.7744140625, 631.8592529296875, 88.7744140625], 'text': 'Characteristics', 'line_num': 1, 'block_num': 36, 'span_num': 201}, {'bbox': [1150.8592529296875, 42.7744140625, 1151.8592529296875, 100.7744140625], 'text': '|', 'line_num': 1, 'block_num': 36, 'span_num': 202}, {'bbox': [92.8592529296875, 117.7744140625, 188.8592529296875, 137.7744140625], 'text': 'Number', 'line_num': 1, 'block_num': 48, 'span_num': 203}, {'bbox': [194.8592529296875, 117.7744140625, 218.8592529296875, 137.7744140625], 'text': 'of', 'line_num': 1, 'block_num': 48, 'span_num': 204}, {'bbox': [224.8592529296875, 117.7744140625, 275.8592529296875, 137.7744140625], 'text': 'cells', 'line_num': 1, 'block_num': 48, 'span_num': 205}, {'bbox': [581.8592529296875, 118.7744140625, 661.8592529296875, 143.7744140625], 'text': '144pcs', 'line_num': 1, 'block_num': 48, 'span_num': 206}, {'bbox': [91.8592529296875, 167.7744140625, 138.8592529296875, 187.7744140625], 'text': 'Size', 'line_num': 2, 'block_num': 48, 'span_num': 207}, {'bbox': [145.8592529296875, 167.7744140625, 169.8592529296875, 187.7744140625], 'text': 'of', 'line_num': 2, 'block_num': 48, 'span_num': 208}, {'bbox': [175.8592529296875, 167.7744140625, 214.8592529296875, 187.7744140625], 'text': 'cell', 'line_num': 2, 'block_num': 48, 'span_num': 209}, {'bbox': [221.8592529296875, 166.7744140625, 282.8592529296875, 189.7744140625], 'text': '(mm)', 'line_num': 2, 'block_num': 48, 'span_num': 210}, {'bbox': [581.8592529296875, 167.7744140625, 661.8592529296875, 188.7744140625], 'text': '166*83', 'line_num': 2, 'block_num': 48, 'span_num': 211}, {'bbox': [91.8592529296875, 220.7744140625, 147.8592529296875, 245.7744140625], 'text': 'Type', 'line_num': 3, 'block_num': 48, 'span_num': 212}, {'bbox': [155.8592529296875, 219.7744140625, 178.8592529296875, 239.7744140625], 'text': 'of', 'line_num': 3, 'block_num': 48, 'span_num': 213}, {'bbox': [184.8592529296875, 219.7744140625, 223.8592529296875, 239.7744140625], 'text': 'cell', 'line_num': 3, 'block_num': 48, 'span_num': 214}, {'bbox': [581.8592529296875, 219.7744140625, 644.8592529296875, 238.7744140625], 'text': 'Mono', 'line_num': 3, 'block_num': 48, 'span_num': 215}, {'bbox': [87.8592529296875, 267.7744140625, 204.8592529296875, 287.7744140625], 'text': 'Thickness', 'line_num': 4, 'block_num': 48, 'span_num': 217}, {'bbox': [212.8592529296875, 267.7744140625, 235.8592529296875, 287.7744140625], 'text': 'of', 'line_num': 4, 'block_num': 48, 'span_num': 218}, {'bbox': [241.8592529296875, 267.7744140625, 299.8592529296875, 293.7744140625], 'text': 'glass', 'line_num': 4, 'block_num': 48, 'span_num': 219}, {'bbox': [306.8592529296875, 266.7744140625, 367.8592529296875, 289.7744140625], 'text': '(mm)', 'line_num': 4, 'block_num': 48, 'span_num': 220}, {'bbox': [581.8592529296875, 267.7744140625, 615.8592529296875, 286.7744140625], 'text': '2.0', 'line_num': 4, 'block_num': 48, 'span_num': 221}, {'bbox': [94.8592529296875, 321.7744140625, 149.8592529296875, 346.7744140625], 'text': 'Type', 'line_num': 5, 'block_num': 48, 'span_num': 223}, {'bbox': [157.8592529296875, 320.7744140625, 181.8592529296875, 340.7744140625], 'text': 'of', 'line_num': 5, 'block_num': 48, 'span_num': 224}, {'bbox': [187.8592529296875, 320.7744140625, 255.8592529296875, 340.7744140625], 'text': 'frame', 'line_num': 5, 'block_num': 48, 'span_num': 225}, {'bbox': [578.8592529296875, 319.7744140625, 688.8592529296875, 339.7744140625], 'text': 'Anodized', 'line_num': 5, 'block_num': 48, 'span_num': 226}, {'bbox': [696.8592529296875, 319.7744140625, 816.8592529296875, 339.7744140625], 'text': 'aluminum', 'line_num': 5, 'block_num': 48, 'span_num': 227}, {'bbox': [824.8592529296875, 319.7744140625, 881.8592529296875, 345.7744140625], 'text': 'alloy', 'line_num': 5, 'block_num': 48, 'span_num': 228}, {'bbox': [94.8592529296875, 369.7744140625, 197.8592529296875, 389.7744140625], 'text': 'Junction', 'line_num': 6, 'block_num': 48, 'span_num': 230}, {'bbox': [206.8592529296875, 369.7744140625, 246.8592529296875, 389.7744140625], 'text': 'box', 'line_num': 6, 'block_num': 48, 'span_num': 231}, {'bbox': [581.8592529296875, 369.7744140625, 630.8592529296875, 388.7744140625], 'text': 'IP68', 'line_num': 6, 'block_num': 48, 'span_num': 232}, {'bbox': [94.8592529296875, 421.7744140625, 140.8592529296875, 441.7744140625], 'text': 'Size', 'line_num': 7, 'block_num': 48, 'span_num': 234}, {'bbox': [147.8592529296875, 421.7744140625, 171.8592529296875, 441.7744140625], 'text': 'of', 'line_num': 7, 'block_num': 48, 'span_num': 235}, {'bbox': [188.8592529296875, 415.7744140625, 266.8592529296875, 452.7744140625], 'text': 'module', 'line_num': 7, 'block_num': 48, 'span_num': 236}, {'bbox': [275.8592529296875, 420.7744140625, 334.8592529296875, 443.7744140625], 'text': '(mm)', 'line_num': 7, 'block_num': 48, 'span_num': 237}, {'bbox': [580.8592529296875, 421.7744140625, 744.8592529296875, 441.7744140625], 'text': '2094*1038*30', 'line_num': 7, 'block_num': 48, 'span_num': 238}, {'bbox': [93.8592529296875, 467.7744140625, 164.8592529296875, 503.7744140625], 'text': 'Weight', 'line_num': 8, 'block_num': 48, 'span_num': 239}, {'bbox': [179.8592529296875, 467.7744140625, 224.8592529296875, 503.7744140625], 'text': '(kg)', 'line_num': 8, 'block_num': 48, 'span_num': 240}, {'bbox': [581.8592529296875, 474.7744140625, 629.8592529296875, 493.7744140625], 'text': '27.5', 'line_num': 8, 'block_num': 48, 'span_num': 241}, {'bbox': [91.8592529296875, 516.7744140625, 312.8592529296875, 550.7744140625], 'text': 'Cables/connectors', 'line_num': 9, 'block_num': 48, 'span_num': 242}, {'bbox': [580.8592529296875, 520.7744140625, 714.8592529296875, 545.7744140625], 'text': '4mm?7,MC4', 'line_num': 9, 'block_num': 48, 'span_num': 243}, {'bbox': [722.8592529296875, 522.7744140625, 854.8592529296875, 548.7744140625], 'text': 'compatible', 'line_num': 9, 'block_num': 48, 'span_num': 244}, {'bbox': [92.8592529296875, 572.7744140625, 172.8592529296875, 598.7744140625], 'text': 'Length', 'line_num': 10, 'block_num': 48, 'span_num': 247}, {'bbox': [180.8592529296875, 572.7744140625, 203.8592529296875, 592.7744140625], 'text': 'of', 'line_num': 10, 'block_num': 48, 'span_num': 248}, {'bbox': [210.8592529296875, 572.7744140625, 275.8592529296875, 592.7744140625], 'text': 'Cabel', 'line_num': 10, 'block_num': 48, 'span_num': 249}, {'bbox': [581.8592529296875, 564.7744140625, 679.8592529296875, 599.7744140625], 'text': 'Portrait:', 'line_num': 10, 'block_num': 48, 'span_num': 250}, {'bbox': [687.8592529296875, 571.7744140625, 894.8592529296875, 595.7744140625], 'text': '+300mm/-300mm', 'line_num': 10, 'block_num': 48, 'span_num': 251}]
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        cropped_img = img.crop(bbox)

        # Applying image filters to enhance quality
        cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(cropped_img)
        cropped_img = enhancer.enhance(2)

        # Scaling the cropped image up for better resolution
        #scale_factor = 2
        #width, height = cropped_img.size
        #cropped_img = cropped_img.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).

    Parameters:
        objects (list of dict): List of detected objects, where each object contains:
            - 'label' (str): The label/category of the object.
            - 'bbox' (list): Bounding box of the object in the form [x_min, y_min, x_max, y_max].
        tokens (list of dict): Tokens corresponding to the objects, with bounding boxes.
        class_thresholds (dict): Dictionary of thresholds for different elements (rows, columns, etc.).

    Returns:
        list of dict: Refined table structures, including rows, columns, headers, spanning cells, and additional info.

    Example:
        >>> objects = [{'label': 'table name', 'score': 0.9929693341255188, 'bbox': [9.332891464233398, 223.05252075195312, 1098.5504150390625, 257.7593994140625]}, {'label': 'table name', 'score': 0.9970752000808716, 'bbox': [16.11467933654785, 77.23551177978516, 1100.463134765625, 203.43641662597656]}, {'label': 'table name', 'score': 0.9991986155509949, 'bbox': [13.618950843811035, 161.23663330078125, 1098.30029296875, 289.1209411621094]}, {'label': 'table name', 'score': 0.99957674741745, 'bbox': [112.92774963378906, 66.88926696777344, 843.1207275390625, 246.92254638671875]}, {'label': 'table name', 'score': 0.9967734217643738, 'bbox': [30.98771858215332, 116.06226348876953, 388.467529296875, 204.75843811035156]}, {'label': 'table name', 'score': 0.9979704022407532, 'bbox': [31.389833450317383, -0.621224045753479, 455.9461669921875, 19.478530883789062]}, {'label': 'table row', 'score': 0.9847909808158875, 'bbox': [10.34100341796875, 144.95721435546875, 1099.7464599609375, 192.7910919189453]}, {'label': 'table name', 'score': 0.9891833066940308, 'bbox': [-10.149672508239746, 5.597719669342041, 1015.55615234375, 42.10792922973633]}, {'label': 'table name', 'score': 0.9975289702415466, 'bbox': [29.789703369140625, -0.5847500562667847, 550.3336791992188, 19.624164581298828]}, {'label': 'table name', 'score': 0.9876828789710999, 'bbox': [6.29730224609375, 1.859797716140747, 1086.9588623046875, 32.25868225097656]}, {'label': 'table name', 'score': 0.993096649646759, 'bbox': [7.232872486114502, 30.671554565429688, 1086.7315673828125, 77.41814422607422]}, {'label': 'table name', 'score': 0.9945861101150513, 'bbox': [11.357348442077637, 211.8690643310547, 1099.85205078125, 256.8695373535156]}, {'label': 'table name', 'score': 0.9995098114013672, 'bbox': [63.324310302734375, 34.65764236450195, 458.2154235839844, 240.3056182861328]}, {'label': 'table name', 'score': 0.9925349950790405, 'bbox': [13.529112815856934, 63.363983154296875, 1095.980224609375, 242.73133850097656]}, {'label': 'table name', 'score': 0.9944188594818115, 'bbox': [13.390470504760742, 46.0146369934082, 1101.8858642578125, 137.34112548828125]}, {'label': 'table name', 'score': 0.9926034808158875, 'bbox': [12.287374496459961, 6.082891464233398, 1097.820556640625, 55.599063873291016]}, {'label': 'table name', 'score': 0.9878190755844116, 'bbox': [34.50221633911133, 71.46537017822266, 367.84600830078125, 251.3552703857422]}, {'label': 'table name', 'score': 0.995058536529541, 'bbox': [9.272468566894531, 228.52963256835938, 1098.42822265625, 267.3962097167969]}, {'label': 'table name', 'score': 0.9891775250434875, 'bbox': [11.104698181152344, -3.345954179763794, 1090.70068359375, 49.23863983154297]}, {'label': 'table name', 'score': 0.9897576570510864, 'bbox': [14.180790901184082, 40.937522888183594, 1093.888916015625, 87.342529296875]}, {'label': 'table name', 'score': 0.9964764714241028, 'bbox': [-34.094886779785156, -3.6862308979034424, 903.8309936523438, 32.30598449707031]}, {'label': 'table name', 'score': 0.9862964153289795, 'bbox': [11.771946907043457, -0.9578020572662354, 614.375732421875, 37.623233795166016]}, {'label': 'table name', 'score': 0.9982982277870178, 'bbox': [6.8338775634765625, 181.0167999267578, 1093.021728515625, 217.35049438476562]}, {'label': 'table name', 'score': 0.994992196559906, 'bbox': [16.35042953491211, 26.800155639648438, 1099.4232177734375, 92.73429107666016]}, {'label': 'table name', 'score': 0.9909458160400391, 'bbox': [15.734076499938965, 64.80786895751953, 1097.9757080078125, 239.45977783203125]}, {'label': 'table name', 'score': 0.9996495246887207, 'bbox': [17.642898559570312, 62.20625305175781, 1085.7939453125, 266.80694580078125]}, {'label': 'table name', 'score': 0.9974072575569153, 'bbox': [12.395439147949219, 46.15504455566406, 425.6272888183594, 237.73214721679688]}, {'label': 'table name', 'score': 0.9979123473167419, 'bbox': [8.39762020111084, 54.40325164794922, 1097.68896484375, 97.3722152709961]}, {'label': 'table name', 'score': 0.9974332451820374, 'bbox': [41.99734878540039, 56.43797302246094, 501.5329284667969, 232.47042846679688]}, {'label': 'table name', 'score': 0.9978711605072021, 'bbox': [13.553812980651855, 31.760196685791016, 1099.503662109375, 85.27610778808594]}, {'label': 'table row', 'score': 0.9913312196731567, 'bbox': [10.501790046691895, 94.37382507324219, 1098.652587890625, 142.47036743164062]}, {'label': 'table name', 'score': 0.9990469813346863, 'bbox': [409.4785461425781, 35.984046936035156, 765.8408203125, 237.5166015625]}, {'label': 'table name', 'score': 0.9985673427581787, 'bbox': [9.430432319641113, 21.4261417388916, 1088.9093017578125, 69.4323501586914]}, {'label': 'table name', 'score': 0.9902048110961914, 'bbox': [-69.7798080444336, -5.146970272064209, 980.1525268554688, 40.784934997558594]}, {'label': 'table name', 'score': 0.9937078952789307, 'bbox': [11.068875312805176, 21.36649513244629, 1090.1934814453125, 60.63591003417969]}, {'label': 'table column', 'score': 0.9176192283630371, 'bbox': [13.768864631652832, 77.57185363769531, 447.6490783691406, 252.7389373779297]}, {'label': 'table name', 'score': 0.9939907789230347, 'bbox': [336.3090515136719, 74.56108093261719, 911.174560546875, 254.66775512695312]}, {'label': 'table name', 'score': 0.9944145679473877, 'bbox': [14.531082153320312, 4.8806328773498535, 1092.4752197265625, 38.150550842285156]}, {'label': 'table name', 'score': 0.9979315996170044, 'bbox': [8.847776412963867, 50.41535949707031, 1093.307861328125, 246.17196655273438]}, {'label': 'table name', 'score': 0.9930064678192139, 'bbox': [8.612357139587402, 211.5151824951172, 1098.48583984375, 246.87692260742188]}, {'label': 'table name', 'score': 0.9797414541244507, 'bbox': [-40.086517333984375, -2.1907854080200195, 772.4074096679688, 38.9447135925293]}, {'label': 'table name', 'score': 0.9900923371315002, 'bbox': [11.281885147094727, -0.1085299551486969, 1092.3001708984375, 39.58427047729492]}, {'label': 'table name', 'score': 0.994779109954834, 'bbox': [-6.3174214363098145, 5.1039719581604, 1069.6708984375, 41.217716217041016]}, {'label': 'table name', 'score': 0.9974905252456665, 'bbox': [68.00594329833984, 12.2147798538208, 605.8729858398438, 90.01889038085938]}, {'label': 'table name', 'score': 0.9993294477462769, 'bbox': [16.133968353271484, 78.49176025390625, 1028.5758056640625, 270.70208740234375]}, {'label': 'table name', 'score': 0.9843303561210632, 'bbox': [133.77020263671875, 142.12411499023438, 990.1473999023438, 202.3194580078125]}, {'label': 'table name', 'score': 0.99936443567276, 'bbox': [78.15201568603516, 39.1330451965332, 425.5627136230469, 260.0869445800781]}, {'label': 'table name', 'score': 0.9991713762283325, 'bbox': [12.702404975891113, 178.3231658935547, 1088.7716064453125, 268.3349914550781]}, {'label': 'table name', 'score': 0.997951328754425, 'bbox': [238.6648712158203, 201.88632202148438, 1103.4068603515625, 242.8484649658203]}, {'label': 'table name', 'score': 0.9995017051696777, 'bbox': [111.46951293945312, 178.14041137695312, 1046.8526611328125, 230.25686645507812]}, {'label': 'table name', 'score': 0.9993220567703247, 'bbox': [38.48290252685547, 144.10720825195312, 1019.7523193359375, 207.79135131835938]}, {'label': 'table name', 'score': 0.9962284564971924, 'bbox': [14.033350944519043, 160.773193359375, 1102.1568603515625, 204.97740173339844]}, {'label': 'table name', 'score': 0.9974615573883057, 'bbox': [16.692289352416992, 138.43179321289062, 1100.9814453125, 270.472412109375]}, {'label': 'table name', 'score': 0.9977734684944153, 'bbox': [15.015267372131348, 28.1357479095459, 1100.4898681640625, 84.47881317138672]}, {'label': 'table name', 'score': 0.9995594620704651, 'bbox': [68.17285919189453, 48.07610321044922, 610.7616577148438, 249.28334045410156]}, {'label': 'table name', 'score': 0.9966784715652466, 'bbox': [192.53814697265625, 53.30458450317383, 442.30133056640625, 255.21739196777344]}, {'label': 'table name', 'score': 0.9936493039131165, 'bbox': [10.903905868530273, 3.4358105659484863, 1089.752685546875, 45.899375915527344]}, {'label': 'table name', 'score': 0.9976301193237305, 'bbox': [444.4818420410156, 90.20330047607422, 698.0900268554688, 237.25421142578125]}, {'label': 'table name', 'score': 0.9956822395324707, 'bbox': [-73.82278442382812, -3.261763095855713, 867.9583740234375, 30.52642059326172]}, {'label': 'table name', 'score': 0.9995468258857727, 'bbox': [52.194480895996094, 157.84275817871094, 1034.8590087890625, 264.75616455078125]}, {'label': 'table name', 'score': 0.9765666723251343, 'bbox': [13.520712852478027, 0.7815062403678894, 1094.54931640625, 52.043312072753906]}, {'label': 'table name', 'score': 0.993767499923706, 'bbox': [4.450331687927246, 3.472249746322632, 1080.3740234375, 36.29048156738281]}, {'label': 'table name', 'score': 0.9993450045585632, 'bbox': [110.10493469238281, 40.12854766845703, 797.8875732421875, 236.0413818359375]}, {'label': 'table name', 'score': 0.9991689920425415, 'bbox': [205.57351684570312, 45.90818405151367, 688.7233276367188, 243.20297241210938]}, {'label': 'table name', 'score': 0.9962026476860046, 'bbox': [21.74557876586914, 56.88501739501953, 418.0995788574219, 157.58450317382812]}, {'label': 'table name', 'score': 0.9994844198226929, 'bbox': [15.909039497375488, 48.27879333496094, 1002.844970703125, 257.11578369140625]}, {'label': 'table row', 'score': 0.9794433116912842, 'bbox': [8.950695991516113, 23.087711334228516, 1100.0347900390625, 90.31526947021484]}, {'label': 'table name', 'score': 0.9973040819168091, 'bbox': [21.79637336730957, 108.83262634277344, 403.2791442871094, 230.66639709472656]}, {'label': 'table name', 'score': 0.9847311973571777, 'bbox': [12.333754539489746, 64.81769561767578, 1100.868896484375, 247.4455108642578]}, {'label': 'table name', 'score': 0.9987745881080627, 'bbox': [11.658702850341797, 120.68810272216797, 1067.40478515625, 183.92799377441406]}, {'label': 'table name', 'score': 0.9900479316711426, 'bbox': [13.951811790466309, 60.41618728637695, 1101.324951171875, 113.836669921875]}, {'label': 'table name', 'score': 0.9995498061180115, 'bbox': [11.236202239990234, 55.78330993652344, 1079.7669677734375, 265.6759948730469]}, {'label': 'table name', 'score': 0.9952043294906616, 'bbox': [16.696239471435547, 67.12281799316406, 1103.327392578125, 255.47171020507812]}, {'label': 'table name', 'score': 0.9971132278442383, 'bbox': [16.035198211669922, 25.760107040405273, 1099.638671875, 84.64054107666016]}, {'label': 'table name', 'score': 0.9789561629295349, 'bbox': [26.275789260864258, -0.41660597920417786, 555.0203247070312, 44.580360412597656]}, {'label': 'table name', 'score': 0.992816686630249, 'bbox': [4.451195240020752, -1.1619906425476074, 708.881591796875, 28.38529396057129]}, {'label': 'table name', 'score': 0.9885631799697876, 'bbox': [19.588970184326172, 84.15290832519531, 392.0896911621094, 232.80860900878906]}, {'label': 'table name', 'score': 0.9936069250106812, 'bbox': [-17.99843406677246, -1.3041770458221436, 1060.64990234375, 30.671403884887695]}, {'label': 'table name', 'score': 0.9901140928268433, 'bbox': [-50.462547302246094, -3.039273500442505, 1028.923583984375, 24.322465896606445]}, {'label': 'table column', 'score': 0.9309638738632202, 'bbox': [456.31561279296875, 79.12478637695312, 1106.7625732421875, 253.17984008789062]}, {'label': 'table name', 'score': 0.9992507100105286, 'bbox': [14.237496376037598, 180.75201416015625, 1098.56591796875, 259.7558898925781]}, {'label': 'table name', 'score': 0.9986827969551086, 'bbox': [443.8627014160156, 144.55819702148438, 1016.9019165039062, 227.47894287109375]}, {'label': 'table name', 'score': 0.9893766641616821, 'bbox': [15.456758499145508, 15.667048454284668, 1072.8472900390625, 68.6160888671875]}, {'label': 'table name', 'score': 0.999049961566925, 'bbox': [9.424158096313477, 218.70858764648438, 1094.74755859375, 273.5520324707031]}, {'label': 'table row', 'score': 0.9919857978820801, 'bbox': [10.206012725830078, 194.15078735351562, 1099.1602783203125, 243.53831481933594]}, {'label': 'table name', 'score': 0.6425030827522278, 'bbox': [11.326173782348633, 15.70832633972168, 1098.9705810546875, 90.05313110351562]}, {'label': 'table name', 'score': 0.9956579208374023, 'bbox': [-8.409605979919434, 1.687553882598877, 979.0283813476562, 59.280269622802734]}, {'label': 'table name', 'score': 0.9948723912239075, 'bbox': [12.288104057312012, 62.43785095214844, 1092.77294921875, 112.85836791992188]}, {'label': 'table name', 'score': 0.9971029162406921, 'bbox': [9.082863807678223, 143.03912353515625, 1096.033447265625, 199.6511688232422]}, {'label': 'table name', 'score': 0.9937455654144287, 'bbox': [-25.462377548217773, 2.348625898361206, 1043.286865234375, 42.616127014160156]}, {'label': 'table name', 'score': 0.9983707070350647, 'bbox': [243.21728515625, 210.238525390625, 1093.7471923828125, 253.1387481689453]}, {'label': 'table name', 'score': 0.9969485402107239, 'bbox': [11.106922149658203, 181.1279754638672, 1099.30078125, 233.08155822753906]}, {'label': 'table name', 'score': 0.9991075396537781, 'bbox': [10.677748680114746, 78.13931274414062, 1094.7044677734375, 272.9703063964844]}, {'label': 'table name', 'score': 0.9896327257156372, 'bbox': [16.424930572509766, 46.8388786315918, 452.3180847167969, 100.20415496826172]}, {'label': 'table name', 'score': 0.9935924410820007, 'bbox': [22.44041633605957, 77.52068328857422, 381.1994323730469, 249.61624145507812]}, {'label': 'table name', 'score': 0.9653909206390381, 'bbox': [475.04437255859375, 82.94808959960938, 858.486328125, 244.72238159179688]}, {'label': 'table name', 'score': 0.9946396946907043, 'bbox': [8.780978202819824, 212.11460876464844, 1100.731201171875, 259.99468994140625]}, {'label': 'table name', 'score': 0.996426522731781, 'bbox': [35.688743591308594, -0.8412526249885559, 628.395751953125, 36.71379470825195]}, {'label': 'table name', 'score': 0.9839897155761719, 'bbox': [2.1961991786956787, -1.0949419736862183, 1078.05908203125, 39.9678840637207]}, {'label': 'table name', 'score': 0.9969570636749268, 'bbox': [-4.962571144104004, -2.4996845722198486, 1022.6210327148438, 44.37287521362305]}, {'label': 'table name', 'score': 0.9880813360214233, 'bbox': [1.5650711059570312, 0.14831750094890594, 1083.5799560546875, 31.156715393066406]}, {'label': 'table name', 'score': 0.9534518718719482, 'bbox': [18.02200698852539, 25.77159309387207, 457.2952880859375, 109.48865509033203]}, {'label': 'table name', 'score': 0.9953786134719849, 'bbox': [67.48809814453125, 70.37427520751953, 777.5850830078125, 121.65461730957031]}, {'label': 'table name', 'score': 0.9494393467903137, 'bbox': [18.975570678710938, 49.46292495727539, 505.8528137207031, 125.5482177734375]}, {'label': 'table name', 'score': 0.9811510443687439, 'bbox': [5.835161209106445, -5.220676898956299, 726.5154418945312, 29.448543548583984]}, {'label': 'table name', 'score': 0.9986374974250793, 'bbox': [39.604557037353516, 66.79187774658203, 957.8365478515625, 248.4901123046875]}, {'label': 'table name', 'score': 0.9992398023605347, 'bbox': [26.024913787841797, 179.0758514404297, 1099.1893310546875, 235.53575134277344]}, {'label': 'table name', 'score': 0.9799265265464783, 'bbox': [26.88844108581543, 91.19145202636719, 415.039794921875, 231.18862915039062]}, {'label': 'table name', 'score': 0.9982782602310181, 'bbox': [73.18451690673828, 55.36553192138672, 580.2047729492188, 234.85446166992188]}, {'label': 'table row header', 'score': 0.7631740570068359, 'bbox': [13.041440963745117, 76.52627563476562, 449.7421875, 250.21580505371094]}, {'label': 'table name', 'score': 0.9981009364128113, 'bbox': [13.396778106689453, 103.76016998291016, 1098.3778076171875, 253.83030700683594]}, {'label': 'table name', 'score': 0.9948768019676208, 'bbox': [14.647380828857422, 2.2951087951660156, 1094.151123046875, 52.37648010253906]}, {'label': 'table name', 'score': 0.9965433478355408, 'bbox': [16.34405517578125, 0.32942503690719604, 904.1132202148438, 32.4370231628418]}, {'label': 'table name', 'score': 0.9950478672981262, 'bbox': [9.618027687072754, -3.478053331375122, 331.3469543457031, 16.209835052490234]}, {'label': 'table name', 'score': 0.9918471574783325, 'bbox': [12.508318901062012, -0.5267461538314819, 1091.0657958984375, 30.779104232788086]}, {'label': 'table name', 'score': 0.9973167777061462, 'bbox': [7.492428302764893, 174.0753936767578, 1096.5260009765625, 216.41896057128906]}, {'label': 'table name', 'score': 0.9951600432395935, 'bbox': [11.056624412536621, 2.260517120361328, 1094.780029296875, 56.37356948852539]}, {'label': 'table name', 'score': 0.9941137433052063, 'bbox': [16.722002029418945, 0.6760401725769043, 1087.7164306640625, 50.1415901184082]}, {'label': 'table name', 'score': 0.9911443591117859, 'bbox': [8.785659790039062, 202.89576721191406, 1098.854736328125, 243.75405883789062]}, {'label': 'table name', 'score': 0.9977052807807922, 'bbox': [8.88630485534668, 187.25030517578125, 415.8813781738281, 248.07762145996094]}, {'label': 'table name', 'score': 0.9994053840637207, 'bbox': [23.528688430786133, 40.45090103149414, 881.9771118164062, 255.8138427734375]}, {'label': 'table name', 'score': 0.995380163192749, 'bbox': [-38.49525451660156, -1.3703835010528564, 956.0086059570312, 38.067466735839844]}, {'label': 'table', 'score': 0.927882194519043, 'bbox': [6.0186896324157715, 23.50992774963379, 1104.891845703125, 261.03057861328125]}, {'label': 'table name', 'score': 0.9994853734970093, 'bbox': [14.546719551086426, 42.08653259277344, 1069.435791015625, 240.76429748535156]}, {'label': 'table name', 'score': 0.9851111769676208, 'bbox': [15.380598068237305, -0.9020041227340698, 1093.7230224609375, 32.468116760253906]}]
        >>> tokens = [{'bbox': [27.3057861328125, 107.758544921875, 145.3057861328125, 133.758544921875], 'text': 'Operating', 'line_num': 12, 'block_num': 48, 'span_num': 259}, {'bbox': [152.3057861328125, 107.758544921875, 346.3057861328125, 133.758544921875], 'text': 'Temperature(°C)', 'line_num': 12, 'block_num': 48, 'span_num': 260}, {'bbox': [518.3057861328125, 103.758544921875, 596.3057861328125, 122.758544921875], 'text': '-40~85', 'line_num': 12, 'block_num': 48, 'span_num': 261}, {'bbox': [27.3057861328125, 157.758544921875, 145.3057861328125, 183.758544921875], 'text': 'Operating', 'line_num': 13, 'block_num': 48, 'span_num': 267}, {'bbox': [153.3057861328125, 157.758544921875, 300.3057861328125, 183.758544921875], 'text': 'humidity(°C)', 'line_num': 13, 'block_num': 48, 'span_num': 268}, {'bbox': [518.3057861328125, 159.758544921875, 573.3057861328125, 178.758544921875], 'text': '5~85', 'line_num': 13, 'block_num': 48, 'span_num': 269}, {'bbox': [26.3057861328125, 207.758544921875, 143.3057861328125, 227.758544921875], 'text': 'Allowable', 'line_num': 14, 'block_num': 48, 'span_num': 275}, {'bbox': [152.3057861328125, 207.758544921875, 196.3057861328125, 227.758544921875], 'text': 'Hail', 'line_num': 14, 'block_num': 48, 'span_num': 276}, {'bbox': [205.3057861328125, 207.758544921875, 260.3057861328125, 227.758544921875], 'text': 'Load', 'line_num': 14, 'block_num': 48, 'span_num': 277}, {'bbox': [518.3057861328125, 210.758544921875, 591.3057861328125, 228.758544921875], 'text': '25mm', 'line_num': 14, 'block_num': 48, 'span_num': 278}, {'bbox': [600.3057861328125, 208.758544921875, 683.3057861328125, 228.758544921875], 'text': 'ice-ball', 'line_num': 14, 'block_num': 48, 'span_num': 279}, {'bbox': [691.3057861328125, 208.758544921875, 742.3057861328125, 228.758544921875], 'text': 'with', 'line_num': 14, 'block_num': 48, 'span_num': 280}, {'bbox': [750.3057861328125, 208.758544921875, 843.3057861328125, 234.758544921875], 'text': 'velocity', 'line_num': 14, 'block_num': 48, 'span_num': 281}, {'bbox': [850.3057861328125, 208.758544921875, 873.3057861328125, 228.758544921875], 'text': 'of', 'line_num': 14, 'block_num': 48, 'span_num': 282}, {'bbox': [880.3057861328125, 208.758544921875, 951.3057861328125, 232.758544921875], 'text': '23m/s', 'line_num': 14, 'block_num': 48, 'span_num': 283}]
        >>> class_thresholds = {'no object': 10, 'table': 0.8, 'table column': 0.6, 'table column header': 0.6, 'table name': 0.6, 'table projected column header': 0.3, 'table projected row header': 0.3, 'table row': 0.6, 'table row header': 0.6, 'table spanning cell': 0.4}
        >>> result = objects_to_structures(objects, tokens, class_thresholds)
        >>> print(result)
        [{'rows': [{'label': 'table row', 'score': 0.8965485692024231, 'bbox': [33.08247756958008, 43.827449798583984, 667.3834838867188, 76.45835876464844], 'column header': False}, {'label': 'table row', 'score': 0.9996610879898071, 'bbox': [33.08247756958008, 76.38275909423828, 667.3834838867188, 108.57452392578125], 'column header': False}, {'label': 'table row', 'score': 0.9998668432235718, 'bbox': [33.08247756958008, 111.3724136352539, 667.3834838867188, 145.10696411132812], 'column header': False}, {'label': 'table row', 'score': 0.9993312358856201, 'bbox': [33.08247756958008, 147.55447387695312, 667.3834838867188, 181.08497619628906], 'column header': False}, {'label': 'table row', 'score': 0.9990942478179932, 'bbox': [33.08247756958008, 183.55348205566406, 667.3834838867188, 215.46286010742188], 'column header': False}], 'columns': [{'label': 'table column', 'score': 0.9999113082885742, 'bbox': [33.08247756958008, 43.827449798583984, 362.59161376953125, 215.46286010742188], 'row header': False, 'column header': False}, {'label': 'table column', 'score': 0.9985477328300476, 'bbox': [371.6669921875, 43.827449798583984, 667.3834838867188, 215.46286010742188], 'row header': False, 'column header': False}], 'column headers': [], 'row headers': [], 'spanning cells': []}]
    """

    tables = [obj for obj in objects if obj['label'] == 'table']
    table_structures = []

    for table in tables:
        table_objects = [obj for obj in objects if iob(obj['bbox'], table['bbox']) >= 0.5]
        table_tokens = [token for token in tokens if iob(token['bbox'], table['bbox']) >= 0.5]

        structure = {}

        # Detect table name and position it appropriately
        table_name = None
        for obj in table_objects:
            if obj['label'] == 'table name':
                table_name = obj
                break

        if table_name:
            table_name_rect = Rect(table_name['bbox'])

            # Check if the table name is part of the first row
            first_row = None
            for row in table_objects:
                if row['label'] == 'table row':
                    first_row_rect = Rect(row['bbox'])
                    if table_name_rect.intersect(first_row_rect).get_area() > 0:
                        first_row = row
                        break

            if first_row:
                # Check if the first row contains only one cell, and the table name spans across it
                if first_row_rect == table_name_rect:
                    structure['table name'] = {
                        'bbox': table_name['bbox'],
                        'position': 'within_row'
                    }
                    table_objects.remove(first_row)  # Remove the first row if it's the table name
                else:
                    structure['table name'] = {
                        'bbox': table_name['bbox'],
                        'position': 'above_row'
                    }
            else:
                structure['table name'] = {
                    'bbox': table_name['bbox'],
                    'position': 'above_row'
                }

        #table_name = [obj for obj in table_objects if obj['label'] == 'table name']
        columns = [obj for obj in table_objects if obj['label'] == 'table column']
        rows = [obj for obj in table_objects if obj['label'] == 'table row']
        column_headers = [obj for obj in table_objects if obj['label'] == 'table column header']
        row_headers = [obj for obj in table_objects if obj['label'] == 'table row header']

        spanning_cells = [obj for obj in table_objects if obj['label'] == 'table spanning cell']
        for obj in spanning_cells:
            obj['projected row header'] = False
            obj['projected column header'] = False

        projected_row_headers = [obj for obj in table_objects if obj['label'] == 'table projected row header']
        for obj in projected_row_headers:
            obj['projected row header'] = True
            obj['projected column header'] = False
        spanning_cells += projected_row_headers

        projected_column_headers = [obj for obj in table_objects if obj['label'] == 'table projected column header']
        for obj in projected_column_headers:
            obj['projected column header'] = True
            obj['projected row header'] = False
        spanning_cells += projected_column_headers

        for obj in rows:
            obj['column header'] = False
            for header_obj in column_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['column header'] = True

        for obj in columns:
            obj['row header'] = False
            for header_obj in row_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['row header'] = True

        # Refine table structures
        rows = postprocess.refine_rows(rows, table_tokens, class_thresholds['table row'])
        columns = postprocess.refine_columns(columns, table_tokens, class_thresholds['table column'])

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])
        column_rect = Rect()
        for obj in columns:
            column_rect.include_rect(obj['bbox'])
        table['row_column_bbox'] = [column_rect[0], row_rect[1], column_rect[2], row_rect[3]]
        table['bbox'] = table['row_column_bbox']

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table['row_column_bbox'])
        rows = postprocess.align_rows(rows, table['row_column_bbox'])

        #structure['table name'] = table_name
        structure['rows'] = rows
        structure['columns'] = columns
        structure['column headers'] = column_headers
        structure['row headers'] = row_headers
        structure['spanning cells'] = spanning_cells

        # Add check to handle tables with both row and column headers
        if len(row_headers) > 0 and len(column_headers) > 0:
            structure['both headers'] = True
        else:
            structure['both headers'] = False

        if len(rows) > 0 and len(columns) > 1:
            structure = refine_table_structure(structure, class_thresholds)

        table_structures.append(structure)

    return table_structures

def structure_to_cells(table_structure, tokens):
    """
    Convert the refined table structure into individual cells, classifying them as header or data cells.
    This function processes table structures, including rows, columns, spanning cells, and headers,
    into a universal cell format that can be further exported to Pandas or CSV formats.

    Each cell is identified by its bounding box, and its classification is determined based on whether
    it intersects with a header bounding box. Spanning cells are also handled, and text content is
    extracted from token spans inside the cell bounding boxes.

    Parameters:
        table_structure (dict): A dictionary containing the structured table data with 'columns', 'rows',
                                 'spanning cells', and 'table name' (optional).
        tokens (list of dict): Tokens corresponding to the detected text, each containing a 'bbox' and text data.

    Returns:
        tuple:
            - list of dict: Cells, each containing the bounding box, column/row information, header flags,
              and associated text data.
            - float: A confidence score indicating how well the page tokens match the table cell structure.

    Example:
            >>> table_structure = {'table name': {'bbox': [30.00767707824707, 278.62286376953125, 1043.2724609375, 319.4166564941406], 'position': 'above_row'}, 'rows': [{'label': 'table row', 'score': 0.9656466245651245, 'bbox': [63, 27, 728, 73], 'column header': False}, {'label': 'table row', 'score': 0.9897612929344177, 'bbox': [63, 91, 728, 117], 'column header': False}, {'label': 'table row', 'score': 0.9910268187522888, 'bbox': [63, 140, 728, 168], 'column header': False}, {'label': 'table row', 'score': 0.9891771674156189, 'bbox': [63, 188, 728, 220], 'column header': False}, {'label': 'table row', 'score': 0.9881728291511536, 'bbox': [63, 237, 728, 267], 'column header': False}, {'label': 'table row', 'score': 0.9894185662269592, 'bbox': [63, 282, 728, 310], 'column header': False}], 'columns': [{'label': 'table column', 'score': 0.9318616986274719, 'bbox': [63, 27, 329, 310], 'row header': True, 'column header': True}, {'label': 'table column', 'score': 0.9548751711845398, 'bbox': [534, 27, 728, 310], 'row header': False, 'column header': True}], 'column headers': [], 'row headers': [{'bbox': [37.289833068847656, 11.574613571166992, 1040.6044921875, 315.38812255859375]}], 'spanning cells': [{'label': 'table spanning cell', 'score': 0.4390396773815155, 'bbox': [37.289833068847656, 11.574613571166992, 1040.6044921875, 75.79998779296875], 'projected row header': False, 'projected column header': False, 'header': False, 'row_numbers': [0], 'column_numbers': [0, 1]}], 'both headers': False, 'table_name': {'bbox': [30.00767707824707, 278.62286376953125, 1043.2724609375, 319.4166564941406], 'position': 'above_row'}}
            >>> tokens = [{'bbox': [65, 28, 221, 73], 'text': 'Packing', 'line_num': 1, 'block_num': 1, 'span_num': 0}, {'bbox': [235, 27, 510, 72], 'text': 'Configuration', 'line_num': 1, 'block_num': 1, 'span_num': 1}, {'bbox': [64, 91, 137, 111], 'text': 'Pieces', 'line_num': 1, 'block_num': 5, 'span_num': 2}, {'bbox': [146, 97, 184, 117], 'text': 'per', 'line_num': 1, 'block_num': 5, 'span_num': 3}, {'bbox': [191, 91, 257, 117], 'text': 'pallet', 'line_num': 1, 'block_num': 5, 'span_num': 4}, {'bbox': [536, 95, 563, 114], 'text': '36', 'line_num': 1, 'block_num': 5, 'span_num': 5}, {'bbox': [64, 142, 110, 162], 'text': 'Size', 'line_num': 1, 'block_num': 5, 'span_num': 6}, {'bbox': [117, 142, 141, 162], 'text': 'of', 'line_num': 1, 'block_num': 5, 'span_num': 7}, {'bbox': [148, 142, 240, 168], 'text': 'packing', 'line_num': 1, 'block_num': 5, 'span_num': 8}, {'bbox': [251, 141, 312, 164], 'text': '(mm)', 'line_num': 1, 'block_num': 5, 'span_num': 9}, {'bbox': [535, 140, 728, 160], 'text': '2130°1140*1190', 'line_num': 1, 'block_num': 5, 'span_num': 10}, {'bbox': [63, 192, 145, 220], 'text': 'Weight', 'line_num': 2, 'block_num': 5, 'span_num': 11}, {'bbox': [152, 192, 176, 212], 'text': 'of', 'line_num': 2, 'block_num': 5, 'span_num': 12}, {'bbox': [183, 192, 275, 220], 'text': 'packing', 'line_num': 2, 'block_num': 5, 'span_num': 13}, {'bbox': [286, 191, 329, 220], 'text': '(kg)', 'line_num': 2, 'block_num': 5, 'span_num': 14}, {'bbox': [537, 188, 591, 207], 'text': '1040', 'line_num': 2, 'block_num': 5, 'span_num': 15}, {'bbox': [64, 241, 137, 261], 'text': 'Pieces', 'line_num': 1, 'block_num': 5, 'span_num': 16}, {'bbox': [146, 247, 184, 267], 'text': 'per', 'line_num': 1, 'block_num': 5, 'span_num': 17}, {'bbox': [190, 241, 303, 262], 'text': 'container', 'line_num': 1, 'block_num': 5, 'span_num': 18}, {'bbox': [537, 237, 578, 256], 'text': '792', 'line_num': 1, 'block_num': 5, 'span_num': 19}, {'bbox': [64, 290, 110, 310], 'text': 'Size', 'line_num': 1, 'block_num': 6, 'span_num': 20}, {'bbox': [117, 290, 141, 310], 'text': 'of', 'line_num': 1, 'block_num': 6, 'span_num': 21}, {'bbox': [147, 290, 260, 310], 'text': 'container', 'line_num': 1, 'block_num': 6, 'span_num': 22}, {'bbox': [534, 282, 604, 302], 'text': '40°HC', 'line_num': 1, 'block_num': 6, 'span_num': 23}]
            >>> cells, confidence_score = structure_to_cells(table_structure, tokens)
            >>> print(cells)
            [{'bbox': [37.289833068847656, 77.70536041259766, 440.29986572265625, 123.46636962890625], 'column_nums': [0], 'row_nums': [1], 'column header': False, 'row header': True, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': 'Pieces per pallet', 'spans': [{'bbox': [64, 91, 137, 111], 'text': 'Pieces', 'line_num': 1, 'block_num': 5, 'span_num': 2}, {'bbox': [146, 97, 184, 117], 'text': 'per', 'line_num': 1, 'block_num': 5, 'span_num': 3}, {'bbox': [191, 91, 257, 117], 'text': 'pallet', 'line_num': 1, 'block_num': 5, 'span_num': 4}]}, {'bbox': [37.289833068847656, 127.3360824584961, 440.29986572265625, 172.37257385253906], 'column_nums': [0], 'row_nums': [2], 'column header': False, 'row header': True, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': 'Size of packing (mm)', 'spans': [{'bbox': [64, 142, 110, 162], 'text': 'Size', 'line_num': 1, 'block_num': 5, 'span_num': 6}, {'bbox': [117, 142, 141, 162], 'text': 'of', 'line_num': 1, 'block_num': 5, 'span_num': 7}, {'bbox': [148, 142, 240, 168], 'text': 'packing', 'line_num': 1, 'block_num': 5, 'span_num': 8}, {'bbox': [251, 141, 312, 164], 'text': '(mm)', 'line_num': 1, 'block_num': 5, 'span_num': 9}]}, {'bbox': [37.289833068847656, 176.07571411132812, 440.29986572265625, 220.73428344726562], 'column_nums': [0], 'row_nums': [3], 'column header': False, 'row header': True, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': 'Weight of packing (kg)', 'spans': [{'bbox': [63, 192, 145, 220], 'text': 'Weight', 'line_num': 2, 'block_num': 5, 'span_num': 11}, {'bbox': [152, 192, 176, 212], 'text': 'of', 'line_num': 2, 'block_num': 5, 'span_num': 12}, {'bbox': [183, 192, 275, 220], 'text': 'packing', 'line_num': 2, 'block_num': 5, 'span_num': 13}, {'bbox': [286, 191, 329, 220], 'text': '(kg)', 'line_num': 2, 'block_num': 5, 'span_num': 14}]}, {'bbox': [37.289833068847656, 223.12820434570312, 440.29986572265625, 266.7409973144531], 'column_nums': [0], 'row_nums': [4], 'column header': False, 'row header': True, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': 'Pieces per container', 'spans': [{'bbox': [64, 241, 137, 261], 'text': 'Pieces', 'line_num': 1, 'block_num': 5, 'span_num': 16}, {'bbox': [146, 247, 184, 267], 'text': 'per', 'line_num': 1, 'block_num': 5, 'span_num': 17}, {'bbox': [190, 241, 303, 262], 'text': 'container', 'line_num': 1, 'block_num': 5, 'span_num': 18}]}, {'bbox': [37.289833068847656, 269.58770751953125, 440.29986572265625, 315.38812255859375], 'column_nums': [0], 'row_nums': [5], 'column header': False, 'row header': True, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': 'Size of container', 'spans': [{'bbox': [64, 290, 110, 310], 'text': 'Size', 'line_num': 1, 'block_num': 6, 'span_num': 20}, {'bbox': [117, 290, 141, 310], 'text': 'of', 'line_num': 1, 'block_num': 6, 'span_num': 21}, {'bbox': [147, 290, 260, 310], 'text': 'container', 'line_num': 1, 'block_num': 6, 'span_num': 22}]}, {'bbox': [447.4510803222656, 77.70536041259766, 1040.6044921875, 123.46636962890625], 'column_nums': [1], 'row_nums': [1], 'column header': False, 'row header': False, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': '36', 'spans': [{'bbox': [536, 95, 563, 114], 'text': '36', 'line_num': 1, 'block_num': 5, 'span_num': 5}]}, {'bbox': [447.4510803222656, 127.3360824584961, 1040.6044921875, 172.37257385253906], 'column_nums': [1], 'row_nums': [2], 'column header': False, 'row header': False, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': '2130°1140*1190', 'spans': [{'bbox': [535, 140, 728, 160], 'text': '2130°1140*1190', 'line_num': 1, 'block_num': 5, 'span_num': 10}]}, {'bbox': [447.4510803222656, 176.07571411132812, 1040.6044921875, 220.73428344726562], 'column_nums': [1], 'row_nums': [3], 'column header': False, 'row header': False, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': '1040', 'spans': [{'bbox': [537, 188, 591, 207], 'text': '1040', 'line_num': 2, 'block_num': 5, 'span_num': 15}]}, {'bbox': [447.4510803222656, 223.12820434570312, 1040.6044921875, 266.7409973144531], 'column_nums': [1], 'row_nums': [4], 'column header': False, 'row header': False, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': '792', 'spans': [{'bbox': [537, 237, 578, 256], 'text': '792', 'line_num': 1, 'block_num': 5, 'span_num': 19}]}, {'bbox': [447.4510803222656, 269.58770751953125, 1040.6044921875, 315.38812255859375], 'column_nums': [1], 'row_nums': [5], 'column header': False, 'row header': False, 'table name': False, 'subcell': False, 'projected row header': False, 'projected column header': False, 'cell text': '40°HC', 'spans': [{'bbox': [534, 282, 604, 302], 'text': '40°HC', 'line_num': 1, 'block_num': 6, 'span_num': 23}]}, {'bbox': [37.289833068847656, 11.574613571166992, 1040.6044921875, 75.79998779296875], 'column_nums': [0, 1], 'row_nums': [0], 'column header': False, 'row header': False, 'projected row header': False, 'projected column header': False, 'table name': {'bbox': [30.00767707824707, 278.62286376953125, 1043.2724609375, 319.4166564941406], 'position': 'above_row'}, 'cell text': 'Packing Configuration', 'spans': [{'bbox': [65, 28, 221, 73], 'text': 'Packing', 'line_num': 1, 'block_num': 1, 'span_num': 0}, {'bbox': [235, 27, 510, 72], 'text': 'Configuration', 'line_num': 1, 'block_num': 1, 'span_num': 1}]}]
            >>> print(confidence_score)
            0.9932551383972168
        """

    #table_name_header = table_structure['table name']
    table_name = table_structure.get('table name', None)
    #table_name = table_structure['table name']
    columns = table_structure['columns']
    rows = table_structure['rows']
    spanning_cells = table_structure['spanning cells']
    cells = []
    subcells = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = Rect(list(column['bbox']))
            row_rect = Rect(list(row['bbox']))
            cell_rect = row_rect.intersect(column_rect)
            #table_name_header = 'table name' in row and row['table name']
            row_header = 'row header' in column and column['row header']
            column_header = 'column header' in row and row['column header']

            # Handle the table name when it's part of the first row
            table_name_flag = False
            if table_name and table_name['position'] == 'within_row' and row_num == 0:
                table_name_flag = True
                if Rect(table_name['bbox']) == cell_rect:
                    # Skip adding this as a regular cell
                    continue

            cell = {'bbox': list(cell_rect), 'column_nums': [column_num], 'row_nums': [row_num],
                    'column header': column_header, 'row header': row_header, 'table name':table_name_flag}


            cell['subcell'] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = Rect(list(spanning_cell['bbox']))
                if (spanning_cell_rect.intersect(cell_rect).get_area()
                    / cell_rect.get_area()) > 0.5:
                    cell['subcell'] = True
                    break

            if cell['subcell']:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell['projected row header'] = False
                cell['projected column header'] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = Rect(list(spanning_cell['bbox']))
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        column_header = True
        row_header = True
        for subcell in subcells:
            subcell_rect = Rect(list(subcell['bbox']))
            subcell_area = subcell_rect.get_area()
            if subcell_area > 0:
                if (subcell_rect.intersect(spanning_cell_rect).get_area() / subcell_area) > 0.5:
                    if cell_rect is None:
                        cell_rect = Rect(list(subcell['bbox']))
                    else:
                        cell_rect.include_rect(Rect(list(subcell['bbox'])))
                    cell_rows = cell_rows.union(set(subcell['row_nums']))
                    cell_columns = cell_columns.union(set(subcell['column_nums']))
                    # By convention here, all subcells must be classified
                    # as header cells for a spanning cell to be classified as a header cell;
                    # otherwise, this could lead to a non-rectangular header region
                    column_header = column_header and subcell.get('column header', False)
                    row_header = row_header and subcell.get('row header', False)
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {'bbox': list(cell_rect), 'column_nums': list(cell_columns), 'row_nums': list(cell_rows),
                    'column header': column_header, 'row header': row_header,
                    'projected row header': spanning_cell['projected row header'],
                    'projected column header': spanning_cell['projected column header'],
                    'table name':table_name}
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = postprocess.slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(dilated_columns[column_num]['bbox']))
        row_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(dilated_rows[row_num]['bbox']))
        cell_rect = column_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(cells, tokens, overlap_threshold=0.001,
                                                               unique_assignment=True, forced_assignment=False)

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell['cell text'] = postprocess.extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell['spans'] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell['spans']:
            min_x_values_by_column[min_column].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_column[max_column].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row['bbox'][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column['bbox'][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column['bbox'][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column['bbox'][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(rows[row_num]['bbox']))
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(columns[column_num]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass

    return cells, confidence_score

def cells_to_csv(cells):
    """
            Converts table cells into a CSV format by organizing them into a structured
            table, where each cell is mapped to its corresponding row and column. The
            headers are flattened and the table is then exported to a CSV .

            Parameters:
                cells (list): List of cell dictionaries, where each cell contains information
                              such as its bounding box, text, row and column numbers, and
                              header classifications.

    """

    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['column header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[:max_header_row + 1, :]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(table_array[max_header_row + 1:, :], index=None, columns=flattened_header)
    #df.to_excel('ex.xlsx', index=None)

    return df.to_csv(index=None)

def cells_to_html(cells):
    """
        Converts a list of table cells into an HTML table representation. Each cell's
        attributes, such as colspan and rowspan, are taken into account when generating
        the table structure.

        Parameters:
            cells (list): List of cell dictionaries, where each cell contains information
                          such as its bounding box, text, row and column numbers, and
                          header classifications.
    """

    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell['column header']:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell['cell text']

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))

def visualize_detected_tables(img, det_tables, out_path):
    """
            Visualizes detected tables on an image and saves the annotated image to a specified path.

            Parameters:
                img: The input image on which tables are detected.
                det_tables (list): A list of dictionaries, each containing information about a detected table,
                                   including its bounding box and label.
                out_path (str): The file path where the output image will be saved.
    """

    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                             label='Table', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return

def visualize_cells(img, cells, out_path):
    """
            Visualizes cells in a detected table on an image and saves the annotated image to a specified path.

            Parameters:
                img (numpy.ndarray): The input image on which cells are detected.
                cells (list): A list of dictionaries, each containing information about a detected cell,
                              including its bounding box and type (e.g., column header, row header, or data cell).
                out_path (str): The file path where the output image will be saved.
        """

    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for cell in cells:
        bbox = cell['bbox']

        if cell['column header']:
            facecolor = (0.9, 0, 0.9)
            edgecolor = (0.9, 0, 0.9)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected row header']:
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['row header']:
            facecolor = (1, 0, 0)
            edgecolor = (1, 0, 0)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected column header']:
            facecolor = (0.49, 0.15, 0.8)
            edgecolor = (0.49, 0.15, 0.8)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['table name']:
            facecolor = (0, 0.8, 0)
            edgecolor = (0, 0.8, 0)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            facecolor = (0.3, 0.74, 0.8)
            edgecolor = (0.3, 0.7, 0.6)
            alpha = 0.3
            linewidth = 2
            hatch = '\\\\\\\\\\\\'

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6),
                             label='Data cell', hatch='\\\\\\\\\\\\', alpha=0.3),
                       Patch(facecolor=(1, 0, 0), edgecolor=(1, 0, 0),
                             label='Row header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.9, 0, 0.9), edgecolor=(0.9, 0, 0.9),
                             label='Column header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Projected row header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0, 0.8, 0), edgecolor=(0, 0.8, 0),
                             label='Table name cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.49, 0.15, 0.8), edgecolor=(0.49, 0.15, 0.8),
                             label='Projected column header cell', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=3)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return

class TableExtractionPipeline(object):
    def __init__(self, det_device=None, str_device=None,
                 det_model=None, str_model=None,
                 det_model_path=None, str_model_path=None,
                 det_config_path=None, str_config_path=None,
                 model_dir=None, in_dir=None, out_dir=None):

        self.det_device = det_device
        self.str_device = str_device
        self.model_dir = model_dir
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.det_class_name2idx = get_class_map('detection')
        self.det_class_idx2name = {v: k for k, v in self.det_class_name2idx.items()}
        self.det_class_thresholds = detection_class_thresholds

        self.str_class_name2idx = get_class_map('structure')
        self.str_class_idx2name = {v: k for k, v in self.str_class_name2idx.items()}
        self.str_class_thresholds = structure_class_thresholds

        if not det_config_path is None:
            with open(det_config_path, 'r') as f:
                det_config = json.load(f)
            det_args = type('Args', (object,), det_config)
            det_args.device = det_device
            self.det_model, _, _ = build_model(det_args)
            print("Detection model initialized.")

            if not det_model_path is None:
                self.det_model.load_state_dict(torch.load(det_model_path,
                                                          map_location=torch.device(det_device)))
                self.det_model.to(det_device)
                self.det_model.eval()
                print("Detection model weights loaded.")
            else:
                self.det_model = None

        if not str_config_path is None:
            with open(str_config_path, 'r') as f:
                str_config = json.load(f)
            str_args = type('Args', (object,), str_config)
            str_args.device = str_device
            self.str_model, _, _ = build_model(str_args)
            print("Structure model initialized.")

            if not str_model_path is None:
                self.str_model.load_state_dict(torch.load(str_model_path,
                                                          map_location=torch.device(str_device)))
                self.str_model.to(str_device)
                self.str_model.eval()
                print("Structure model weights loaded.")
            else:
                self.str_model = None

    def __call__(self, page_image, page_tokens=None):
        return self.extract(self, page_image, page_tokens)

    def detect(self, img, tokens=None, out_objects=True, out_crops=False, crop_padding=10):
        out_formats = {}
        if self.det_model is None:
            print("No detection model loaded.")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = detection_transform(img)

        # Run input image through the model
        outputs = self.det_model([img_tensor.to(self.det_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.det_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not out_crops:
            return out_formats

        # Crop image and tokens for detected table
        if out_crops:
            tables_crops = objects_to_crops(img, tokens, objects, self.det_class_thresholds,
                                            padding=crop_padding)
            out_formats['crops'] = tables_crops

        return out_formats

    def recognize(self, img, tokens=None, out_objects=False, out_cells=False,
                  out_html=False, out_csv=False):
        out_formats = {}
        if self.str_model is None:
            print("No structure model loaded.")
            return out_formats

        if not (out_objects or out_cells or out_html or out_csv):
            print("No output format specified")
            return out_formats

        # Transform the image how the model expects it
        img = enhance_image(img).convert('RGB')
        img_tensor = structure_transform(img)

        # Run input image through the model
        outputs = self.str_model([img_tensor.to(self.str_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not (out_cells or out_html or out_csv):
            return out_formats

        # Further process the detected objects so they correspond to a consistent table
        tables_structure = objects_to_structures(objects, tokens, self.str_class_thresholds)

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [structure_to_cells(structure, tokens)[0] for structure in tables_structure]
        if out_cells:
            out_formats['cells'] = tables_cells
        if not (out_html or out_csv):
            return out_formats

        # Convert cells to HTML
        if out_html:
            tables_htmls = [cells_to_html(cells) for cells in tables_cells]
            out_formats['html'] = tables_htmls

        # Convert cells to CSV, including flattening multi-row column headers to a single row
        if out_csv:
            tables_csvs = [cells_to_csv(cells) for cells in tables_cells]
            out_formats['csv'] = tables_csvs
            out_formats['xlsx'] = tables_csvs
            #tables_csvs_df = pd.DataFrame(tables_csvs)
            #writer = pd.ExcelWriter(os.path.join(), engine='xlsxwriter')
            #tables_csvs_df.to_excel(writer, sheet_name='1', index=False)
            #writer.save()

        return out_formats

    def extract(self, img, tokens=None, out_objects=True, out_crops=False, out_cells=False,
                out_html=False, out_csv=False, crop_padding=10):

        detect_out = self.detect(img, tokens=tokens, out_objects=False, out_crops=True,
                                 crop_padding=crop_padding)
        cropped_tables = detect_out['crops']

        extracted_tables = []
        for table in cropped_tables:
            img = table['image']
            tokens = table['tokens']

            extracted_table = self.recognize(img, tokens=tokens, out_objects=out_objects,
                                             out_cells=out_cells, out_html=out_html, out_csv=out_csv)
            extracted_table['image'] = img
            extracted_table['tokens'] = tokens
            extracted_tables.append(extracted_table)

        return extracted_tables

    def perform_final_step(self, excel_folder_path, pdf_folder_path):
        """
        This function performs the final step of extracting and structuring values
        from Excel files within a folder. It uses regex patterns to extract data
        and returns it as a dictionary. This is the primary function for processing
        all the Excel files in the specified folder.

        Parameters:
        - excel_folder_path: str, the path to the folder containing Excel files to be processed.
        - pdf_folder_path: str, the path to the folder containing corresponding PDF files.

        Returns:
        - A dictionary with the extracted values from the Excel files.

    """

        print("Performing final Step")
        all_files = self.fs_folder(
            path_to_excel_folder=excel_folder_path,
            path_to_pdf_folder=pdf_folder_path
        )
        self.extracted_values = all_files
        print("final Step completed")

    def fs_folder(
            self,
            path_to_excel_folder: str,
            path_to_pdf_folder: str
    ) -> dict:
        """
        This internal function processes all the Excel files in the specified folder.
        It iterates through the Excel files, extracts values using the `fs_file` function,
        and stores them in a dictionary where the key is the filename (without extension)
        and the value is the extracted data.

        Parameters:
        - path_to_excel_folder: str, the path to the folder containing Excel files to be processed.
        - path_to_pdf_folder: str, the path to the folder containing corresponding PDF files.

        Returns:
        - A dictionary where the keys are filenames (without the `.xlsx` extension)
          and the values are the extracted data from those files.

        """

        list_of_files = get_list_of_files_with_ext(
            path_to_folder=path_to_excel_folder,
            ext=".xlsx",
            verbose=True
        )

        all_files_extracted = {}

        # Go through all the files one by one and get the
        # values that were extracted
        for file in list_of_files:
            #filename = str(basename(file)).rsplit(sep=".")[0]
            filename = os.path.splitext(file)[0]

            # Just run the function for now
            extracted_vals = self.fs_file(
                path_to_excel_file=file,
                path_to_pdf_file=filename + ".pdf"
            )

            all_files_extracted[filename] = extracted_vals

        return all_files_extracted

    def extract_data_with_llm(row_data, field_description):
        """
            This function uses a language model (like GPT-4) to extract a specific field
            from a given row of data based on a description provided in the prompt.

            It constructs a prompt by combining the field description and the row data,
            queries the LLM for the relevant information, and returns the extracted value.

            Parameters:
            - row_data (str): The data from which a field value needs to be extracted.
            - field_description (str): A description of the field to be extracted.

            Returns:
            - str: The extracted value as returned by the LLM.

            Example:
            >>> row_data = "The total revenue for Q1 is $50,000 and the profit is $12,000."
            >>> field_description = "total revenue"
            >>> extracted_value = extract_data_with_llm(row_data, field_description)
            >>> print(extracted_value)
            "$50,000"
        """

        # Format your prompt for the LLM
        prompt = f"Extract the {field_description} from the following data: {row_data}"

        # Query GPT-4 (or any other LLM)
        response = openai.Completion.create(
            engine="gpt-4",  # You can also use 'gpt-3.5-turbo'
            prompt=prompt,
            max_tokens=100,
            temperature=0  # Low temperature for deterministic responses
        )

        # Extract the text from the LLM response
        extracted_value = response.choices[0].text.strip()
        return extracted_value

    def save_to_excel(
            self,
            path: str = "extracted_data.xlsx"
    ) -> None:
        """
        This is the path where the excel file containing the extracted
        values will be saved.

        Args:
            path:
                This is the path to the excel file where the extracted
                values will be saved.
        """

        print()

        print("Saving to excel")
        print("---------------")

        final_list = []

        for name, prop_type in self.extracted_values.items():

            # Get the values
            thermal = prop_type.get("thermal")
            electrical = prop_type.get("electrical")
            electrical_nmot = prop_type.get("electrical_nmot")
            pack = prop_type.get("pack")

            year = prop_type.get("misc").get("year")

            length = prop_type.get("mech").get("length")
            width = prop_type.get("mech").get("width")
            height = prop_type.get("mech").get("height")

            # Assuming equal lengths of extracted arrays of values
            if electrical is not None:
                value_count = []

                for value_type, value_list in electrical.items():
                    if value_list is not None:
                        value_count.append(len(value_list))

                # most_freq_count = mode(value_count)
                value_count_cleaned = np.nan_to_num(value_count, nan=np.nan)

                # Filter out NaN values
                value_count_cleaned = value_count_cleaned[~np.isnan(value_count_cleaned)]

                if len(value_count_cleaned) == 0:
                    most_freq_count = np.nan
                else:
                    most_freq_count = int(stats.mode(value_count_cleaned)[0])
            else:
                most_freq_count = 1
            """elif electrical_nmot is not None:
                value_count = []

                for value_type, value_list in electrical_nmot.items():
                    if value_list is not None:
                        value_count.append(len(value_list))

                # most_freq_count = mode(value_count)
                value_count_cleaned = np.nan_to_num(value_count, nan=np.nan)

                # Filter out NaN values
                value_count_cleaned = value_count_cleaned[~np.isnan(value_count_cleaned)]

                if len(value_count_cleaned) == 0:
                    most_freq_count = np.nan
                else:
                    most_freq_count = int(stats.mode(value_count_cleaned)[0])"""


            if math.isnan(most_freq_count):
                curr_file_list = [[]]
            else:
                name = name.split("\\")[-1]
                curr_file_list = [[name] * most_freq_count]
                curr_file_list.append([year] * most_freq_count)

                curr_file_list.append([length] * most_freq_count)
                curr_file_list.append([width] * most_freq_count)
                curr_file_list.append([height] * most_freq_count)

                # Adding electrical properties
                elec_prop_types = ["eff", "pmpp", "vmpp", "impp", "voc", "isc", "ff"]

                for prop in elec_prop_types:

                    if electrical is not None:
                        vals = electrical.get(prop)
                    else:
                        vals = None

                    if vals is None:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        if len(vals) == most_freq_count:
                            curr_file_list.append(vals)
                        elif len(vals) < most_freq_count:
                            # Append empty strings to make the length equal to most_freq_count
                            curr_file_list.append(vals + [""] * (most_freq_count - len(vals)))
                        elif len(vals) > most_freq_count:
                            # Trim the list to match most_freq_count
                            curr_file_list.append(vals[:most_freq_count])

                # adding electrival propertoes in Nominal module operating temperature
                elec_nmot_prop_types = ["eff_nmot", "pmpp_nmot", "vmpp_nmot", "impp_nmot", "voc_nmot", "ise_nmot"]
                for prop in elec_nmot_prop_types:
                    if electrical_nmot is not None:
                        vals = electrical_nmot.get(prop)
                    else:
                        vals = None

                    if vals is None:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        if len(vals) == most_freq_count:
                            curr_file_list.append(vals)
                        elif len(vals) < most_freq_count:
                            # Append empty strings to make the length equal to most_freq_count
                            curr_file_list.append(vals + [""] * (most_freq_count - len(vals)))
                        elif len(vals) > most_freq_count:
                            # Trim the list to match most_freq_count
                            curr_file_list.append(vals[:most_freq_count])

                # Adding thermal properties
                thermal_prop_types = ["isc", "pmpp", "voc"]

                for prop in thermal_prop_types:

                    if thermal is not None:
                        vals = thermal.get(prop)
                    else:
                        vals = None

                    if vals is None or len(vals) == 0:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        curr_file_list.append(vals * most_freq_count)

                        """# Adding thermal properties
                        thermal_prop_types = ["isc", "pmpp", "voc"]

                        for prop in thermal_prop_types:

                            if thermal is not None:
                                vals = thermal.get(prop)
                            else:
                                vals = None

                            if vals is None:
                                curr_file_list.append([""] * most_freq_count)
                            else:
                                curr_file_list.append(vals * most_freq_count)"""

                #Adding packaging properties
                pack_prop_types = ["pcs_pallet", "pallet_containter"]

                for prop in pack_prop_types:
                    if pack is not None:
                        vals = pack.get(prop)
                    else:
                        vals = None

                    if vals is None or len(vals) == 0:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        curr_file_list.append(vals * most_freq_count)

            # Transpose the list
            curr_file_list = list(map(list, zip(*curr_file_list)))

            final_list.extend(curr_file_list)

        # Create a dataframe
        final_df = pd.DataFrame(final_list,
                                columns=[
                                    "name",
                                    "year",
                                    "length",
                                    "width",
                                    "height",
                                    "E/eff",
                                    "E/pmpp",
                                    "E/vmpp",
                                    "E/impp",
                                    "E/voc",
                                    "E/isc",
                                    "E/ff",
                                    "ENMOT/eff",
                                    "ENMOT/pmpp",
                                    "ENMOT/vmpp",
                                    "ENMOT/impp",
                                    "ENMOT/voc",
                                    "ENMOT/isc",
                                    "T/isc",
                                    "T/pmpp",
                                    "T/voc",
                                    "P/pcs_pallet",
                                    "P/pallet_container"
                                ]
                                )


        final_df.to_excel(path)
        print("final excel created")
        # print(final_df)

        # Write to the excel file
        #with pd.ExcelWriter(path=path, mode='w') as writer:
            #final_df.to_excel(writer,index=False)

        #wb = openpyxl.Workbook()
        #ws = wb.active
        #ws.append(final_df)
        #wb.save(path)

    def fs_file(
            self,
            path_to_excel_file: str,
            path_to_pdf_file: str
    ) -> dict:
        """
        This is an internal function that will perform the final step
        for the excel file specified and return a dictionary of items
        that were extracted. It will combine the electrical and thermal
        properties together into a single dictionary and just keep the
        values that were extracted and not the rows where they were
        found on.
        """

        # Load the yaml file that contains all the patterns for
        # detecting the correct columns and the values
        filepath = os.path.join(self.model_dir, "patterns.yaml")
        with open(filepath, "r", encoding='utf-8') as stream:
            try:
                patterns = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)


        # Get type specific patterns
        elec_patterns_STC = patterns.get("electrical")
        elec_patterns_NMOT = patterns.get("electrical_nmot")
        therm_patterns = patterns.get("temperature")
        mech_patterns = patterns.get("mechanical")
        pack_patterns = patterns.get("packaging")

        curr_ds = Datasheet(
            path_to_pdf=path_to_pdf_file,
            path_to_excel=path_to_excel_file,
            path_to_clf=self.model_dir + "/nb_classifier_latest.pickle",
            path_to_vec=self.model_dir + "/vectoriser_latest.pickle"
        )

        curr_ds.extract_electrical_props(patterns=elec_patterns_STC)
        elec_extracted = curr_ds.extracted_elec

        curr_ds.extract_electrical_props_nmot(patterns=elec_patterns_NMOT)
        elec_extracted_nmot = curr_ds.extracted_elec_nmot

        curr_ds.extract_temp_props(patterns=therm_patterns)
        therm_extracted = curr_ds.extracted_temp

        curr_ds.extract_pack_props(patterns=pack_patterns)
        pack_extracted = curr_ds.extracted_pack

        curr_ds.extract_mech_props(patterns=mech_patterns)
        mech_extracted = curr_ds.extracted_mech

        curr_ds.extract_misc_props()
        misc_extracted = curr_ds.extracted_misc

        if elec_extracted is not None:
            for key, item in elec_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                elec_extracted[key] = vals

        if elec_extracted_nmot is not None:
            for key, item in elec_extracted_nmot.items():
                vals = item.get("vals")

                if vals is not None:
                    vals=[str(x) for x in vals]
                elec_extracted_nmot[key] = vals


        if therm_extracted is not None:
            for key, item in therm_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                therm_extracted[key] = vals

        if pack_extracted is not None:
            for key, item in pack_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                pack_extracted[key] = vals

        """if mech_extracted is not None:
            for key, item in mech_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                mech_extracted[key] = vals"""

        return {
            "electrical": elec_extracted,
            "electrical_nmot": elec_extracted_nmot,
            "thermal": therm_extracted,
            "pack": pack_extracted,
            "mech": mech_extracted,
            "misc": misc_extracted
        }

def output_result(key, val, args, img, img_file):
    """
           Process detected objects (tables, cells, crops) and save them as structured output (CSV, HTML),
           and optionally visualize the results.

           Parameters:
           - key: The type of result being processed like ('objects', 'crops', 'cells', 'html').
           - val: The detected data like table objects, cropped tables, or cells.
           - args: Arguments from the parser, including 'out_dir' (output directory),
                               'verbose' (to print output), and 'visualize' (to save visualizations).
           - img: The input image used to visualize tables/cells.
           - img_file: The file name of the input image to generate corresponding output files.
        """
    if key == 'objects':
        if args.verbose:
            print(val)
        out_file = img_file.replace(".jpg", "_objects.json")
        with open(os.path.join(args.out_dir, out_file), 'w') as f:
            json.dump(val, f)
        if args.visualize:
            out_file = img_file.replace(".jpg", "_fig_tables.jpg")
            out_path = os.path.join(args.out_dir, out_file)
            visualize_detected_tables(img, val, out_path)
    elif not key == 'image' and not key == 'tokens':
        for idx, elem in enumerate(val):
            if key == 'crops':
                for idx, cropped_table in enumerate(val):
                    out_img_file = img_file.replace(".jpg", "_table_{}.jpg".format(idx))

                    # Ensuring that the output directory exists
                    os.makedirs(args.out_dir, exist_ok=True)

                    cropped_table['image'].save(os.path.join(args.out_dir,
                                                             out_img_file), 'JPEG', quality=95, optimize=True)
                    out_words_file = out_img_file.replace(".jpg", "_words.json")
                    with open(os.path.join(args.out_dir, out_words_file), 'w') as f:
                        json.dump(cropped_table['tokens'], f)
            elif key == 'cells':
                out_file = img_file.replace(".jpg", "_{}_objects.json".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w') as f:
                    json.dump(elem, f)
                if args.verbose:
                    print(elem)
                if args.visualize:
                    out_file = img_file.replace(".jpg", "_fig_cells.jpg")
                    out_path = os.path.join(args.out_dir, out_file)
                    visualize_cells(img, elem, out_path)
            elif key == "csv":
                out_file = img_file.replace(".jpg", "_{}.csv".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w') as f:
                    if elem is not None:
                        f.write(elem)
                if args.verbose:
                    print(elem)
            elif key == "xlsx":
                # Parse the CSV content
                parsed_data = list(csv.reader(StringIO(elem), delimiter=',', quotechar='"'))

                # Convert the parsed data into a DataFrame
                df = pd.DataFrame(parsed_data)

                out_file =  os.path.join(args.out_dir ,img_file.replace(".jpg", "_{}.xlsx".format(idx)))

                df.to_excel(out_file, index=False, header=False)
                '''out_file = img_file.replace(".jpg", "_{}.xlsx".format(idx))
                wb = openpyxl.Workbook()
                ws = wb.active

                if elem is not None:
                    # Split the string into rows based on newline characters
                    rows = elem.split('\r\n')
                    for row in rows:
                        # Split each row into cells based on comma separation
                        cells = row.split(',')
                        ws.append(cells)
                    wb.save(os.path.join(args.out_dir, out_file))'''




def get_list_of_files_with_ext(
    path_to_folder: str,
    ext: str,
    randomise: bool = False,
    verbose: bool = True
    ) -> list:
    """
    This function will go through all the files in the given
    folder and make a list of files with the provided extension.
    This can be used, for example, to filter out the required
    files in the folder.

    Parameters:
        path_to_folder:
            This is the path to folder that will be scanned
            for the required files.

        ext:
            This is the extension of the files that will be
            selected from the folder.

        randomise:
            If this flag is set to True, then the list of files
            will be shuffled before being returned.

        verbose:
            If this flag is set to True, then this function will
            display the information from the folder.

    Returns:
        list_of_files:
            This is the list of files in the provided
            directory (folder) that matches the extension
            provided. It contains the full path to the files
            not just the name of the files.
    """

    list_of_files = []

    # Evaluate all files in the directory
    for file in listdir(path_to_folder):

        # Skip the hidden files
        # In linux and macOS, the hidden files start
        # with '.'
        if not file.startswith('.'):

            # Get the files with the specified extension
            if file.endswith(ext):
                full_path = join(path_to_folder, file)
                list_of_files.append(full_path)

    if verbose:
        print()
        print("Looking for " + ext + " files in folder: " + path_to_folder)
        print()
        print("Total " + ext + " files found: " + str(len(list_of_files)))

    # Shuffle the list of files captured
    if randomise:
        random.shuffle(list_of_files)

    return list_of_files


def merge_tables(excel_files_folder, output_folder, pdf_files):
    for pdf_file in pdf_files:
        # Get the base name of the PDF file
        base_name = os.path.splitext(pdf_file)[0]

        # Initialize a list to store DataFrames
        dfs = []

        # Iterate through Excel files in the same folder
        for filename in os.listdir(excel_files_folder):
            if filename.startswith(base_name) and filename.endswith('.xlsx'):
                # Read the Excel file
                df = pd.read_excel(os.path.join(excel_files_folder, filename))

                # Append the DataFrame to combined_df
                dfs.append(df)

        # Concatenate all DataFrames in dfs list
        if dfs:
            combined_file_path = os.path.join(output_folder, base_name + '.xlsx')
            with pd.ExcelWriter(combined_file_path) as writer:
                for idx, df in enumerate(dfs):
                    # Write each DataFrame to a separate worksheet
                    sheet_name = f'{base_name}_{idx}'
                    df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)

        else:
            print("No matching files found for:", pdf_file)


def main():
    #setting the tesseract path for OCR
    #pytesseract.pytesseract.tesseract_cmd = r'D:\Users\swa86085\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    global pdf_files_dir, image_dir, pdf_files
    #pytesseract.pytesseract.tesseract_cmd = r'S:\23502\2\216_PVM\Aktuell\01_Orga\21631_MAS_TeamModulbewertung\03_Arbeitsordner\Swathi_Thiruvengadam\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Admin\Desktop\thesis\swathi-thiruvengadam\Tesseract-OCR\tesseract.exe'

    #ssh
    #pytesseract.pytesseract.tesseract_cmd = r'/net/s/23502/2/280_PVM/Aktuell/01_Orga/23131_MAS_TeamModulbewertung/03_Arbeitsordner/Swathi_Thiruvengadam/Tesseract-OCR/tesseract.exe'
    start_time = time.time()

    args = get_args()
    print(args.__dict__)
    print('-' * 100)

    # create an output directory if it does not exists
    if not args.out_dir is None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Create inference pipeline
    print("Creating inference pipeline")
    pipe = TableExtractionPipeline(det_device=args.detection_device,
                                   str_device=args.structure_device,
                                   det_config_path=args.detection_config_path,
                                   det_model_path=args.detection_model_path,
                                   str_config_path=args.structure_config_path,
                                   str_model_path=args.structure_model_path,
                                   model_dir = args.model_dir,
                                   in_dir = args.in_dir,
                                   out_dir = args.out_dir)

    # Load images
    # convert the pdf file to images
    if args.mode == 'detect' or args.mode == 'extract':
        pdf_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(('.pdf'))]

        # Directory for storing intermediate images
        image_dir = os.path.join(args.in_dir, '..', 'images')
        os.makedirs(image_dir, exist_ok=True)

        for pdf in pdf_files:
            poppler_path = 'C:/Users/Admin/anaconda3/envs/tables-detr/Library/bin/'
            pdf_path = os.path.join(args.in_dir, pdf)
            pdf_image = convert_from_path(pdf_path, dpi=300)
            image_base_path = os.path.join(image_dir, os.path.splitext(pdf)[0])
            #for idx in range(len(pdf_image)):
                #pdf_image[idx].save(image_path + '_' + str(idx + 1) + '.jpg', 'JPEG')
            for idx, image in enumerate(pdf_image):
                #enhanced_image = enhance_image(image)
                image_path = f"{image_base_path}_{idx + 1}.jpg"
                image.save(image_path, 'JPEG')

    elif args.mode == 'recognize':
        pdf_files_dir = os.path.join(args.in_dir, '..', 'pdf')
        pdf_files = [f for f in os.listdir(pdf_files_dir) if f.lower().endswith(('.pdf'))]
        image_dir = os.path.join(args.in_dir, '..', 'images')
        os.makedirs(image_dir, exist_ok=True)

        for pdf in pdf_files:
            pdf_path = os.path.join(pdf_files_dir, pdf)
            pdf_image = convert_from_path(pdf_path, dpi=300)
            image_path = os.path.join(image_dir, os.path.splitext(pdf)[0])
              #for idx in range(len(pdf_image)):
                #pdf_image[idx].save(image_path + '_' + str(idx + 1) + '.jpg', 'JPEG')
            for idx, image in enumerate(pdf_image):
                #enhanced_image = enhance_image(image)
                image_path = f"{image_path}_{idx + 1}.jpg"
                image.save(image_path, 'JPEG')


    # create the OCR words
    if args.mode == 'detect' or args.mode == 'extract':
        #words_dir = "../inferences/detectionwords"
        words_dir = os.path.join("..", "inferences", "detectionwords")
        # OCR program to extract the words for all the image files present in our folder
        extract_words_from_images(image_dir, words_dir)
    elif args.mode == 'recognize':
        words_dir = os.path.join("..", "inferences", "words")
        image_dir = args.in_dir


    # List only image files in the input folder
    img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    num_files = len(img_files)
    random.shuffle(img_files)

    # note : change directory paths to make more sense
    img_input_folder = os.path.join(image_dir, '..', 'detectionOutput')
    words_output_folder = os.path.join(image_dir, '..', 'words')

    for count, img_file in enumerate(img_files):
        print("({}/{})".format(count + 1, num_files))
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        #latest
        #img_path = image_dir +'/' +  img_file
        #img = Image.open(img_path).convert('RGB')
        print("Image loaded.")

        if not words_dir is None:
            #tokens_path = words_dir + "/" + img_file.replace(".jpg", "_words.json")
            tokens_path = os.path.join(words_dir, img_file.replace(".jpg", "_words.json"))
            with open(tokens_path, 'r') as f:
                tokens = json.load(f)

                # Handle dictionary format
                if type(tokens) is dict and 'words' in tokens:
                    tokens = tokens['words']

                # 'tokens' is a list of tokens
                # Need to be in a relative reading order
                # If no order is provided, use current order
                for idx, token in enumerate(tokens):
                    if not 'span_num' in token:
                        token['span_num'] = idx
                    if not 'line_num' in token:
                        token['line_num'] = 0
                    if not 'block_num' in token:
                        token['block_num'] = 0
        else:
            tokens = []

        if args.mode == 'recognize':
            extracted_table = pipe.recognize(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                             out_html=args.html, out_csv=args.csv)
            print("Table(s) recognized.")

            for key, val in extracted_table.items():
                output_result(key, val, args, img, img_file)

            # merge into same file
            merge_tables(args.out_dir, pdf_files_dir, pdf_files)


        if args.mode == 'detect':
            detected_tables = pipe.detect(img, tokens, out_objects=args.objects, out_crops=args.crops)
            print("Table(s) detected.")

            for key, val in detected_tables.items():
                output_result(key, val, args, img, img_file)

        if args.mode == 'extract':
            extracted_tables = pipe.extract(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                            out_html=args.html, out_csv=args.csv,
                                            crop_padding=args.crop_padding)
            print("Table(s) extracted.")

            for table_idx, extracted_table in enumerate(extracted_tables):
                for key, val in extracted_table.items():
                    output_result(key, val, args, extracted_table['image'],
                                  img_file.replace('.jpg', '_{}.jpg'.format(table_idx)))

            # merge into same file
            merge_tables(args.out_dir, args.in_dir, pdf_files)

    if args.mode == 'detect':
        extract_words_from_images(img_input_folder, words_output_folder)

    if args.mode == 'recognize':
        pipe.perform_final_step(pdf_files_dir, pdf_files_dir)
        pipe.save_to_excel(path=os.path.join(pdf_files_dir, "extracted_tables.xlsx"))

    if args.mode == 'extract':
        pipe.perform_final_step(args.in_dir, args.in_dir)
        pipe.save_to_excel(path=os.path.join(args.in_dir, "extracted_tables.xlsx"))

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time taken by Table-Transformer to extract tabular data is : {total_time} seconds")

if __name__ == "__main__":
    main()