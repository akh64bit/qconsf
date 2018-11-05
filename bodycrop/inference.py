'''
Some is based on Ildoo Kim's code (https://github.com/ildoonet/tf-openpose) and https://gist.github.com/alesolano/b073d8ec9603246f766f9f15d002f4f4
and derived from the OpenPose Library (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.core.framework import graph_pb2
import urllib3
import certifi

from common import estimate_pose, crop_image, draw_humans

import time


def print_time(message, start):
    print(message, "{:10.4f}".format(time.time() - start))
    return time.time()


class SmartBodyCrop:
    initialized = False
    
    def __init__(self, model_url):
        self.model_url = model_url

    def read_img(self, imgpath, width, height):
        val_img = Image.open(imgpath)
        val_img = val_img.resize((width, height))
        val_img = np.asarray(val_img)
        val_img = val_img.reshape([1, height, width, 3])
        val_img = val_img.astype(float)
        val_img = val_img * (2.0 / 255.0) - 1.0

        return val_img
    
    def _download_model(self):
        # check if the model is a ref to local file path
        if type(self.model_url) is str:
            if not self.model_url.startswith('http'):
                return self.model_url
            
        start = time.time()
        local_model_path = '/tmp/optimized_openpose.pb'
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED', 
            ca_certs=certifi.where(),
            headers={
                'Accept': 'application/octet-stream',
                'Content-Type': 'application/octet-stream'
            })
        urllib3.disable_warnings()
        
        r = http.request('GET', self.model_url, 
                         preload_content=False,
                         retries=urllib3.Retry(5, redirect=5))

        with open(local_model_path, 'wb') as out:
            while True:
                data = r.read(32)
                if not data:
                    break
                out.write(data)
        
        r.release_conn()
        print_time("model downloaded in :", start)
        return local_model_path
        
    def _download_image(self, image):
        start = time.time()
        headers = {}
        image_url = image
        local_image_path = '/tmp/image'
        if type(image) is dict:
            headers = image.get('headers')
            image_url = image.get('uri')
        # check if the image is a local file path
        if type(image) is str:
            if not image.startswith('http'):
                return image
            
        http = urllib3.PoolManager(
            cert_reqs = 'CERT_REQUIRED', 
            ca_certs = certifi.where(),
            headers = headers)
        urllib3.disable_warnings()
        
        r = http.request('GET', image_url, 
                         preload_content=False,
                         retries=urllib3.Retry(5, redirect=5))

        with open(local_image_path, 'wb') as out:
            while True:
                data = r.read(32)
                if not data:
                    break
                out.write(data)
        
        r.release_conn()
        print_time("image downloaded in :", start)
        return local_image_path

    def load_graph_def(self):
        start = time.time()
        
        local_model_path = self._download_model()

        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        with open(local_model_path, 'rb') as f: 
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        start = print_time("model imported in :", start)
        start = time.time()

        SmartBodyCrop.initialized = True

    def infer(self, image, upper_body, lower_body):
        start = time.time()

        imgpath = self._download_image(image)
        img1 = Image.open(imgpath)
        img1 = np.asarray(img1)
        input_width, input_height = img1.shape[0], img1.shape[1]

        image = self.read_img(imgpath, input_width, input_height)
        start = print_time("image loaded in: ", start)

        if not SmartBodyCrop.initialized:
            print("Loading the model...")
            self.load_graph_def()

        with tf.Session() as sess:
            inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
            heatmaps_tensor = tf.get_default_graph().get_tensor_by_name(
                'Mconv7_stage6_L2/BiasAdd:0')
            pafs_tensor = tf.get_default_graph().get_tensor_by_name(
                'Mconv7_stage6_L1/BiasAdd:0')

            heatMat, pafMat = sess.run(
                [heatmaps_tensor, pafs_tensor], feed_dict={inputs: image})

            start = print_time("tf session executed in: ", start)

            humans = estimate_pose(heatMat[0], pafMat[0])
            start = print_time("pose estimated in: ", start)

            img, crop_coordinates = crop_image(imgpath, humans, upper_body, lower_body)
            start = print_time("image cropped in: ", start)

            sess.close()
            return img, crop_coordinates

    def detect_parts(self, image):
        start = time.time()

        imgpath = self._download_image(image)
        img1 = Image.open(imgpath)
        img1 = np.asarray(img1)
        input_width, input_height = img1.shape[0], img1.shape[1]

        image = self.read_img(imgpath, input_width, input_height)
        start = print_time("image loaded in: ", start)

        if not SmartBodyCrop.initialized:
            print("Loading the model...")
            self.load_graph_def()

        with tf.Session() as sess:
            inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
            heatmaps_tensor = tf.get_default_graph().get_tensor_by_name(
                'Mconv7_stage6_L2/BiasAdd:0')
            pafs_tensor = tf.get_default_graph().get_tensor_by_name(
                'Mconv7_stage6_L1/BiasAdd:0')

            heatMat, pafMat = sess.run(
                [heatmaps_tensor, pafs_tensor], feed_dict={inputs: image})

            start = print_time("tf session executed in: ", start)

            humans = estimate_pose(heatMat[0], pafMat[0])
            start = print_time("pose estimated in: ", start)
            # display
            image_h, image_w = img1.shape[:2]
            img1_raw = Image.open(imgpath)
            img1 = draw_humans(img1_raw, humans)

            scale = 480.0 / image_h
            newh, neww = 480, int(scale * image_w + 0.5)

            return img1
