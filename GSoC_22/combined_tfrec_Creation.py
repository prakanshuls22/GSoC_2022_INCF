import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np 
import matplotlib.pyplot as plt
import json
from PIL import Image
import os

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "device": bytes_feature(example["device"]),
        "screen_h": int64_feature(example["screen_h"]),
        "screen_w": int64_feature(example["screen_w"]),
        "face_valid": int64_feature(example["face_valid"]),
        "face_x": int64_feature(example["face_x"]),
        "face_y": int64_feature(example["face_y"]),
        "face_w": int64_feature(example["face_w"]),
        "face_h": int64_feature(example["face_h"]),
        "leye_x": int64_feature(example["leye_x"]),
        "leye_y": int64_feature(example["leye_y"]),
        "leye_w": int64_feature(example["leye_w"]),
        "leye_h": int64_feature(example["leye_h"]),
        "reye_x": int64_feature(example["reye_x"]),
        "reye_y": int64_feature(example["reye_y"]),
        "reye_w": int64_feature(example["reye_w"]),
        "reye_h": int64_feature(example["reye_h"]),
        "dot_xcam": float_feature(example["dot_xcam"]),
        "dot_y_cam": float_feature(example["dot_y_cam"]),
        "dot_x_pix": float_feature(example["dot_x_pix"]),
        "dot_y_pix": float_feature(example["dot_y_pix"]),
        "reye_x1": int64_feature(example["reye_x1"]),
        "reye_y1": int64_feature(example["reye_y1"]),
        "reye_x2": int64_feature(example["reye_x2"]),
        "reye_y2": int64_feature(example["reye_y2"]),
        "leye_x1": int64_feature(example["leye_x1"]),
        "leye_y1": int64_feature(example["leye_y1"]),
        "leye_x2": int64_feature(example["leye_x2"]),
        "leye_y2": int64_feature(example["leye_y2"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "device": tf.io.FixedLenFeature([], tf.string),
        "screen_h": tf.io.FixedLenFeature([], tf.int64),
        "screen_w": tf.io.FixedLenFeature([], tf.int64),
        "face_valid": tf.io.FixedLenFeature([], tf.int64),
        "face_x": tf.io.FixedLenFeature([], tf.int64),
        "face_y": tf.io.FixedLenFeature([], tf.int64),
        "face_w": tf.io.FixedLenFeature([], tf.int64),
        "face_h": tf.io.FixedLenFeature([], tf.int64),
        "leye_x": tf.io.FixedLenFeature([], tf.int64),
        "leye_y": tf.io.FixedLenFeature([], tf.int64),
        "leye_w": tf.io.FixedLenFeature([], tf.int64),
        "leye_h": tf.io.FixedLenFeature([], tf.int64),
        "reye_x": tf.io.FixedLenFeature([], tf.int64),
        "reye_y": tf.io.FixedLenFeature([], tf.int64),
        "reye_w": tf.io.FixedLenFeature([], tf.int64),
        "reye_h": tf.io.FixedLenFeature([], tf.int64),
        "dot_xcam": tf.io.FixedLenFeature([], tf.float32),
        "dot_y_cam": tf.io.FixedLenFeature([], tf.float32),
        "dot_x_pix": tf.io.FixedLenFeature([], tf.float32),
        "dot_y_pix": tf.io.FixedLenFeature([], tf.float32),
        "reye_x1": tf.io.FixedLenFeature([], tf.int64),
        "reye_y1": tf.io.FixedLenFeature([], tf.int64),
        "reye_x2": tf.io.FixedLenFeature([], tf.int64),
        "reye_y2": tf.io.FixedLenFeature([], tf.int64),
        "leye_x1": tf.io.FixedLenFeature([], tf.int64),
        "leye_y1": tf.io.FixedLenFeature([], tf.int64),
        "leye_x2": tf.io.FixedLenFeature([], tf.int64),
        "leye_y2": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example

main_path = os.getcwd() #path of unzipped gazetrack file
# main_path = "/Users/prakanshulsaxena/GSoC/task_1_GSoC_dadded/gazetrack_gsplit/" #path of unzipped gazetrack file
set_of_data = [main_path+"/gazetrack/train/",main_path+"/gazetrack/test/",main_path+"/gazetrack/val/"]

# main_id = '03467' #individual id whose tfrec you want to create

list_tfrecs = []

list_of_ids = ['01866',
 '02459',
 '01816',
 '03004',
 '03253',
 '01231',
 '00503',
 '02152',
 '02015',
 '01046']

for i in list_of_ids:

    save_path = '/home/sheenu22/projects/def-skrishna/shared/' + i + '.tfrec'
    list_tfrecs.append(save_path)

    with tf.io.TFRecordWriter(save_path) as writer: #path of tfrec you want to save

        for k in set_of_data:

            curr_json_path = k + "meta/"
            curr_imgf_path = k + "images/"

            for j in os.listdir(curr_imgf_path):

                img_path = curr_imgf_path + j
                img_id = j.split('.j')[0]
                p_id = j.split('__')[0]

                if(i == str(p_id)):

                    json_path = curr_json_path + img_id + '.json'
                    json_file = json.load(open(json_path))
                    image = tf.io.decode_jpeg(tf.io.read_file(img_path))
                    example = create_example(image, img_path, json_file)
                    writer.write(example.SerializeToString())


dataset = tf.data.TFRecordDataset(list_tfrecs)

filename = '/home/sheenu22/projects/def-skrishna/shared/combined_individuals.tfrec'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(dataset)






