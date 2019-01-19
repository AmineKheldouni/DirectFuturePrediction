import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../Mask_RCNN_matterport/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def display_medkit_mask(image, mask, regions, class_ids, scores, CLASS_NAMES):
    """Display the given image and the top few class masks."""
    to_display = []
    to_display.append(image)
    # Pick top prominent classes in this image
    chosen_class_ids = []
    if len(class_ids) > 0:
        for i in range(len(class_ids)):
            if not class_ids[i] in chosen_class_ids and class_ids[i] in [6,8]:
                chosen_class_ids.append(class_ids[i])

    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in chosen_class_ids]
    top_ids = [v[0] for v in sorted(zip(chosen_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    class_id = top_ids[0] if 0 < len(top_ids) else -1
    # Pull masks of instances belonging to the same class.
    m = mask[:, :, np.where(class_ids == class_id)[0]].astype('float32')
    m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)

    for i in range(len(regions)):
        m[regions[i][0]:regions[i][2],regions[i][1]:regions[i][3]] = \
        np.sign(m[regions[i][0]:regions[i][2],regions[i][1]:regions[i][3]]) * scores[i]

    # to_display.append(m)
    # visualize.display_images(to_display, titles=['simulator image', 'medkit mask'], cols= 2, cmap="Blues_r")
    return m

def predict_segmentation(image, model):
    results = model.detect([image], verbose=0)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                CLASS_NAMES, r['scores'])
    m = display_medkit_mask(image, r['masks'], r['rois'], r['class_ids'], r['scores'], CLASS_NAMES)

    return m

#
file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# for i in range(10):
i = 6
image = skimage.io.imread(IMAGE_DIR+'/VizDoom/vizdoom' + str(i) + '.png')

# Run detection
segmentation = predict_segmentation(image, model)
