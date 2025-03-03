
import cv2
import numpy as np
import os
import csv
import re
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

import tensorflow as tf
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


class GestureDetails:


    def __init__(self, gesture_Id, gesture_name, output_label):
        self.gesture_Id = gesture_Id
        self.gesture_name = gesture_name
        self.output_label = output_label


class GestureFeature:


    def __init__(self, gesture_detail: GestureDetails, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature


def extract_feature(folder_path, input_file, mid_frame_counter):
    middle_image = cv2.imread(frameExtractor(folder_path + input_file, folder_path + "frames/", mid_frame_counter), cv2.IMREAD_GRAYSCALE)
    print(middle_image)
    feature_extracted = HandShapeFeatureExtractor.extract_feature(HandShapeFeatureExtractor.get_instance(),middle_image)
    return feature_extracted


def get_gesture_by_file_name(gesture_file_name):
    for x in gesture_data:
        if x.output_label == gesture_file_name.split('_')[0]:
            return x
    return None


# a list to containg all gestures and thier details (Id, name, label)
gesture_data = [GestureDetails("Num0", "0", "0"), GestureDetails("Num1", "1", "1"),
                GestureDetails("Num2", "2", "2"), GestureDetails("Num3", "3", "3"),
                GestureDetails("Num4", "4", "4"), GestureDetails("Num5", "5", "5"),
                GestureDetails("Num6", "6", "6"), GestureDetails("Num7", "7", "7"),
                GestureDetails("Num8", "8", "8"), GestureDetails("Num9", "9", "9"),
                GestureDetails("FanDown", "Decrease Fan Speed", "10"),
                GestureDetails("FanOn", "FanOn", "11"), GestureDetails("FanOff", "FanOff", "12"),
                GestureDetails("FanUp", "Increase Fan Speed", "13"),
                GestureDetails("LightOff", "LightOff", "14"), GestureDetails("LightOn", "LightOn", "15"),
                GestureDetails("SetThermo", "SetThermo", "16")
                ]

featureVectorList = []
train_data_path = "traindata/"
count = 0
for file in os.listdir(train_data_path):
    if not file.startswith('frames') and file.endswith('mp4'):
        featureVectorList.append(GestureFeature(get_gesture_by_file_name(file),
                                                extract_feature(train_data_path, file, count)))
        count = count + 1


def gesture_detection(gesture_folder_path, gesture_file_name, mid_frame_counter):

    video_feature = extract_feature(gesture_folder_path, gesture_file_name, mid_frame_counter)

    find_match = False
    num_mutations = 0
    gesture_detail: GestureDetails = GestureDetails("", "", "")
    while not find_match and num_mutations < 5:
        similarity = 1
        position = 0
        index = 0
        for featureVector in featureVectorList:
            cosine_similarity = tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature, axis=-1)
            if cosine_similarity > similarity:
                similarity = cosine_similarity
                position = index
            index = index + 1

        gesture_detail = featureVectorList[position].gesture_detail
        find_match = True
        if not find_match:
            num_mutations = num_mutations + 1
    return gesture_detail

test_data_path = "test/"
test_count = 0
with open('Results.csv', 'w', newline='') as results_file:
    data_writer = csv.writer(results_file)

    for test_file in os.listdir(test_data_path):
        if not test_file.startswith('frames') and test_file.endswith('mp4'):
            recognized_gesture_detail = gesture_detection(test_data_path, test_file, test_count)
            test_count = test_count + 1

            data_writer.writerow({recognized_gesture_detail.output_label})