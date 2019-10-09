import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
# isess.run(tf.global_variables_initializer())
tf.global_variables_initializer().run()#初始化变量
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    image_pre_def=isess.run(image_pre,feed_dict={img_input:img})
    print('img_pre.shape=',image_pre_def.shape)
    print('bboxes=',isess.run(bboxes_pre,feed_dict={img_input:img}))
    print('bbox_img=',isess.run(bbox_img,feed_dict={img_input:img}))

    # 图片预处理
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    print('rimg.shape=',rimg.shape)
    print('rpredictions.len=',len(rpredictions))
    print('rlocalistaions.len=',len(rlocalisations))
    print('rbbox_img=',rbbox_img)

    # Get classes and bboxes from the net outputs.从输出中获取类别和范围
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    print('rclasses=',rclasses)
    print('rscores=',rscores)
    print('rbboxes.shape=',rbboxes.shape)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)#按准确率降序排序

    print('rclasses=', rclasses)
    print('rscores=', rscores)
    print('rbboxes.shape=', rbboxes.shape)

    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)#去重
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!

    print('rclasses=', rclasses)
    print('rscores=', rscores)
    print('rbboxes.shape=', rbboxes.shape)
    print('rbboxes=',rbboxes)


    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
# 测试的文件夹
path = '../demo/'
image_names = sorted(os.listdir(path))
# 文件夹中的第几张图，-1代表最后一张
img = mpimg.imread(path + image_names[-2])
rclasses, rscores, rbboxes = process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)