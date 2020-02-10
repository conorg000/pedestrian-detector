import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
"""
Script to load a TF inference graph and perform evaluation on an image file
Returns bounding box coordinates, confidence, inference time
"""

# VARIABLES TO CHANGE
# Change to local env path (wherever you cloned ped-detector)
PATH = '/content/'
# Change to desired jpeg image
image_path = PATH + 'ped-detector/MOT16-01_000001.jpg'

# Path to frozen detection graph. This is the actual fine-tuned model that is used for the object detection.
PATH_TO_CKPT = PATH + 'ped-detector/fine-tuned-model/frozen_inference_graph.pb'
# For pre-trained model
#PATH_TO_CKPT = PATH + 'ped-detector/pre-trained-model/frozen_inference_graph.pb'
# Path to label map
PATH_TO_LABELS = PATH + 'ped-detector/annotations/ped_label_map.pbtxt'
# Detect just one class
NUM_CLASSES = 1

def infer_img(image_path, confidence=0.3):
  """
  Performs pedestrian object detection on an image
  Returns bounding box coordinates (where confidence is over confidence%)

  Input:
  image_path (string): path to .jpg file (change paths at top of this script as necessary)
  confidence (float): confidence level (function only returns detections above this level)

  Output:
  results (list of tuples): list containing tuples of detections
                            each tuple is of format (confidence, [ymin, xmin, ymax, xmax])
                            bounding box coordinates are absolute (not relative)
  time (float): inference time in seconds
  """
  # Load the inference graph
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

  # Match labels with names and their index
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # Resize input image and direct to new path
  img = Image.open(image_path)
  img.thumbnail((600, 600))
  image_path = image_path[:-4] + "_resized.jpg"
  img.save(image_path)

  # Start tf detection graph
  with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
          # Load tensors
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
          detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Load image
          image = Image.open(image_path)
          # Make into array
          (im_width, im_height) = image.size
          image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Start inference timer
          t0 = time.time()
          # Get results from inference
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Stop inference timer
          t1 = time.time()

  scores = scores.tolist()
  scores = [float(x) for x in scores[0]]
  scores = list(filter(lambda x: x > 0.3, scores))
  num = len(scores)
  inf_time = float(t1-t0)
  bbs = boxes.tolist()[0][:num]
  results = list(zip(scores, bbs))
  for result in results:
    # [ymin, xmin, ymax, xmax]
    result[1][0] *= im_height
    result[1][1] *= im_width
    result[1][2] *= im_height
    result[1][3] *= im_width
  return (results, inf_time)

coordinates, inf_time = infer_img(image_path)
print(coordinates)
print(inf_time)
