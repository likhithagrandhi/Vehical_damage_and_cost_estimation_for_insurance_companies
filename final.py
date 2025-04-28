import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from pathlib import Path

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model
import mrcnn.model as modellib
from mrcnn.model import log
import cv2
import custom,custom_1
import imgaug,h5py,IPython

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
custom_WEIGHTS_PATH = "mask_rcnn_coco.h5"  # TODO: update this path for best performing iteration weights
config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "custom/")

# Load dataset
dataset = custom_1.CustomDataset()
dataset.load_custom(custom_DIR, "train")

# Must call before using the dataset
dataset.prepare()
print('Here is the dataset: ', dataset)
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))




def done():
    image_id = random.choice(dataset.image_ids)
    # image_id = (app/image_path)
    print('Image id is : ', image_id)
    image = dataset.load_image(image_id) 
    # filepath = './uploads/1.jpg'
    # image = cv2.imread(filepath)
    # image_id = 1
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    print('Class ids: ', class_ids)
    # Display image and instances
    visualize.save_image(image, 'result', bbox, mask, class_ids, class_names = ['damages'])