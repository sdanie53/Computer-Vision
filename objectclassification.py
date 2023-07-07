#!/usr/env/bin python3

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 
import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    #local_model_path="~/Downloads/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8"
    # Check if the path exists
    # if not os.path.exists(local_model_path):
    #     raise ValueError('Model path does not exist')
    
    model_dir = pathlib.Path(model_dir)/"saved_model"

    #The function tf.saved_model.load() is used to load models that were exported using the tf.saved_model.save() function or the SavedModel APIs in TensorFlow 2.x. It loads the entire model back, including its weights and the computation.
    model = tf.saved_model.load(str(model_dir))

    #When you save a model using tf.saved_model.save, you can specify multiple signatures, or ways to run the model. For example, you might have a signature for training, and another signature for inference (predictions). Each signature would have different inputs and outputs.
    #The serving default signature is usually the one that's used for inference. In the context of this object detection script, it defines that the model takes in images and outputs detected objects, their bounding boxes, and other related information.
    model = model.signatures['serving_default']

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

cap = cv2.VideoCapture(0)

while True:
    ret, image_np = cap.read()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3] ::: (batch_size, height, width, channels) ::: batch_size = 1 and channels = 3 for RGB images or 1 for grayscale
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(detection_model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
