import os
import re
import glob
import skimage.io
from itertools import cycle
import numpy as np
import tensorflow as tf
from libs import vgg16
from PIL import Image

LEARNING_RATE = 0.02
BATCH_SIZE = 4
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3]
SKIP_STEP = 10
N_EPOCHS = 600
CKPT_DIR = './Checkpoints/'
IMG_DIR = './Images/'
GRAPH_DIR = './Graphs/'
TRAINING_SET_DIR= './dataset/training/'
GROUNDTRUTH_SET_DIR='./dataset/groundtruth/'
VALIDATION_SET_DIR='./dataset/validation/'
METRICS_SET_DIR='./dataset/metrics/'
TRAINING_DIR_LIST = []


ADVERSARIAL_LOSS_FACTOR = 0.5
PIXEL_LOSS_FACTOR = 1.0
STYLE_LOSS_FACTOR = 1.0
SMOOTH_LOSS_FACTOR = 0.0001
metrics_image = skimage.io.imread(METRICS_SET_DIR+'lung-300ma0006.jpg').astype('float32')


def initialize(sess):
    saver = tf.train.Saver(max_to_keep=5)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver

def get_training_dir_list():
    training_list = [d[1] for d in os.walk(TRAINING_SET_DIR)]
    global TRAINING_DIR_LIST
    TRAINING_DIR_LIST = training_list[0]
    return TRAINING_DIR_LIST

def load_next_training_batch():
    batch = next(pool_training)
    return batch

def load_next_groundtruth_batch():
    batch = next(pool_groundtruth)
    return batch

def load_validation():
    filelist = sorted(glob.glob(VALIDATION_SET_DIR + '/*.jpg'), key=alphanum_key)
    validation = np.array([np.array(skimage.io.imread(fname).astype('float32')) for fname in filelist])
    if validation.ndim != 4:
        validation = skimage.color.gray2rgb(validation)
    return validation

def training_dataset_init():
    filelist = sorted(glob.glob(TRAINING_SET_DIR + '/*.jpg'), key=alphanum_key)
    batch_training = np.array([np.array(skimage.io.imread(fname).astype('float32')) for fname in filelist])

    filelist_t = sorted(glob.glob(GROUNDTRUTH_SET_DIR + '/*.jpg'), key=alphanum_key)
    batch_groundtruth = np.array([np.array(skimage.io.imread(fname).astype('float32')) for fname in filelist_t])

    if batch_groundtruth.ndim != 4:
        batch_training = skimage.color.gray2rgb(batch_training)
        batch_groundtruth = skimage.color.gray2rgb(batch_groundtruth)
    return batch_training,batch_groundtruth

def getcycle(batch_training,batch_groundtruth):
    batch_training = split(batch_training, BATCH_SIZE)
    batch_groundtruth = split(batch_groundtruth, BATCH_SIZE)
    global pool_training
    global pool_groundtruth
    pool_training = cycle(batch_training)

    pool_groundtruth = cycle(batch_groundtruth)


def imsave(filename, image):
    skimage.io.imsave(IMG_DIR+filename+'.jpg', image)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def RGB_TO_BGR(img):
    img_channel_swap = img[..., ::-1]
    # img_channel_swap_1 = tf.reverse(img, axis=[-1])
    return img_channel_swap

def gray_to_rgb(img):
    # img_channel = np.concatenate((img,img,img),axis=-1)
    img_channel = tf.concat([img,img,img],axis=-1)
    return img_channel

def get_pixel_loss(target,prediction):
    pixel_difference = target - prediction
    pixel_loss = tf.nn.l2_loss(pixel_difference)
    return pixel_loss

def get_style_layer_vgg16(image):
    net = vgg16.get_vgg_model()
    style_layer = 'conv2_2/conv2_2:0'
    feature_transformed_image = tf.import_graph_def(
        net['graph_def'],
        name='vgg',
        input_map={'images:0': image},return_elements=[style_layer])
    feature_transformed_image = (feature_transformed_image[0])
    return feature_transformed_image

def get_style_loss(target,prediction):
    feature_transformed_target = get_style_layer_vgg16(target)
    feature_transformed_prediction = get_style_layer_vgg16(prediction)
    feature_count = tf.shape(feature_transformed_target)[3]
    style_loss = tf.reduce_sum(tf.square(feature_transformed_target-feature_transformed_prediction))
    style_loss = style_loss/tf.cast(feature_count**3, tf.float32)
    return style_loss

def get_smooth_loss(image):
    batch_count = tf.shape(image)[0]
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    horizontal_normal = tf.slice(image, [0, 0, 0,0], [batch_count, image_height, image_width-1,3])
    horizontal_one_right = tf.slice(image, [0, 0, 1,0], [batch_count, image_height, image_width-1,3])
    vertical_normal = tf.slice(image, [0, 0, 0,0], [batch_count, image_height-1, image_width,3])
    vertical_one_right = tf.slice(image, [0, 1, 0,0], [batch_count, image_height-1, image_width,3])
    smooth_loss = tf.nn.l2_loss(horizontal_normal-horizontal_one_right)+tf.nn.l2_loss(vertical_normal-vertical_one_right)
    return smooth_loss

def standardization(img):
    result=np.zeros((img.shape[0],img.shape[1],img.shape[2],img.shape[3]),dtype="float32")
    for i in range(0,img.shape[0]):
        temp = img[i,:,:,:]
        mean = np.mean(temp,axis=(0,1))
        stddev = np.std(temp,axis=(0,1))
        NumElements =np.array([1.0/np.sqrt(img.shape[1]*img.shape[2]),1.0/np.sqrt(img.shape[1]*img.shape[2]),1.0/np.sqrt(img.shape[1]*img.shape[2])])   #3通道使用这个
        d = np.array((stddev,NumElements))
        adjusted_stddev =d.max(axis=0)
        result[i,:,:,:]=(temp-mean)/adjusted_stddev
    return result

def standardization_1(img):
    result=np.zeros((img.shape[0],img.shape[1],img.shape[2],img.shape[3]),dtype="float32")
    for i in range(0,img.shape[0]):
        temp = img[i,:,:,:]
        max = np.max(temp,axis=(0,1))
        min = np.min(temp,axis=(0,1))
        k = (1-(-1))/(max - min)
        result[i,:,:,:] = -1 + k*(temp - min)
    return result

def average_image(img):
    if img.shape[2]==1:
        return img
    img = np.sum(img,-1)/3
    return img

