'''
Some is based on Ildoo Kim's code (https://github.com/ildoonet/tf-openpose) and https://gist.github.com/alesolano/b073d8ec9603246f766f9f15d002f4f4
and derived from the OpenPose Library (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''

import tensorflow as tf
import numpy as np
from PIL import Image

from common import estimate_pose, crop_image, draw_humans

def read_img(imgpath,width,height):
    val_img = Image.open(imgpath)
    val_img=val_img.resize((width,height))
    val_img=np.asarray(val_img)
    val_img = val_img.reshape([1, height, width, 3])
    val_img = val_img.astype(float)
    val_img = val_img * (2.0 / 255.0) - 1.0
    
    return val_img

def infer(imgpath,upper_body,lower_body):
    img1 = Image.open(imgpath)
    img1=np.asarray(img1)
    input_width,input_height=img1.shape[0],img1.shape[1]
    tf.reset_default_graph()
    
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    with open('models/optimized_openpose.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

    image = read_img(imgpath, input_width, input_height)

    with tf.Session() as sess:
        heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
            inputs: image
        })

        heatMat, pafMat = heatMat[0], pafMat[0]
        humans = estimate_pose(heatMat, pafMat)
        img=crop_image(imgpath,humans,upper_body,lower_body)
        return img
    
    
def detect_parts(imgpath):
    img1_raw = Image.open(imgpath)
    img1=np.asarray(img1_raw)
    input_width,input_height=img1.shape[0],img1.shape[1]
    tf.reset_default_graph()
    
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    with open('models/optimized_openpose.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

    image = read_img(imgpath, input_width, input_height)

    with tf.Session() as sess:
        heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
            inputs: image
        })

        heatMat, pafMat = heatMat[0], pafMat[0]
        humans = estimate_pose(heatMat, pafMat)
         # display
        image_h, image_w = img1.shape[:2]
        img1 = draw_humans(img1_raw, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        return img1
