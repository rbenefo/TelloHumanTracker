import numpy as np
import os
import urllib.request
import sys
import tarfile
import tensorflow as tf
import tarfile

import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# from object_detection.utils import visualization_utils as vis_util


class PersonDetector:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        curr_dir = os.getcwd()

        if not os.path.isdir(curr_dir+"/"+model_name):
            #add functionality to check of model is already downloaded.
            print("Grabbing model!")
            base_url = 'http://download.tensorflow.org/models/object_detection/'
            model_file = model_name + '.tar.gz'
            # model_dir = tf.keras.utils.get_file(
            #         fname=curr_dir+"/"+model_name, 
            #         origin=base_url + model_file,
            #         extract=True)
            
            tar_path, _ = urllib.request.urlretrieve(base_url+model_file)
            model_tar = tarfile.open(tar_path)
            model_tar.extractall() 
            model_tar.close()

            os.remove(tar_path)

        else:
            print("Already got the model!")

        model_dir = pathlib.Path(curr_dir+"/"+model_name)/"saved_model"
        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']
        return model
    
    def infer(self, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]
        output_dict = self.model(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        boxes, scores = self.identify_humans(output_dict)
        return boxes, scores
    
    def identify_humans(self, output_dict):
        boxes = np.squeeze(output_dict["detection_boxes"])
        scores = np.squeeze(output_dict["detection_scores"])
        classes = np.squeeze(output_dict["detection_classes"])

        indices = np.argwhere(classes == 1)
        boxes = np.squeeze(boxes[indices])
        scores = np.squeeze(scores[indices])

        return boxes, scores


