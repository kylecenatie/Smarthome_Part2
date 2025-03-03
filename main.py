import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import csv
import re as regex
from sklearn.utils import Bunch  # Import Bunch

import frameextractor as fe
import handshape_feature_extractor as hfe


def extract_image_feature(location, input_file, mid_frame_counter):
    """
    Extracts features from the middle frame of a video file.

    Args:
        location (str): Directory containing the video file.
        input_file (str): Name of the video file.
        mid_frame_counter (int): Frame number to extract.

    Returns:
        numpy.ndarray: Extracted features from the middle frame.
    """
    middle_image = cv2.imread(
        fe.frameExtractor(location + input_file, location + "frames/", mid_frame_counter),
        cv2.IMREAD_GRAYSCALE
    )
    response = hfe.HandShapeFeatureExtractor.get_instance().extract_feature(middle_image)
    return response


def find_gesture_by_filename(gesture_file_name):
    """
    Finds a gesture by parsing the gesture key from the filename.

    Args:
        gesture_file_name (str): Name of the file containing the gesture.

    Returns:
        Bunch: Gesture details if found, otherwise None.
    """
    return next((x for x in gesture_details if x.gesture_key == gesture_file_name.split('_')[0]), None)


def find_gesture_by_name(lookup_gesture_name):
    """
    Finds a gesture by matching the gesture name.

    Args:
        lookup_gesture_name (str): Name of the gesture to find.

    Returns:
        Bunch: Gesture details if found, otherwise None.
    """
    return next((x for x in gesture_details if x.gesture_name.replace(" ", "").lower() == lookup_gesture_name.lower()), None)


def identify_gesture_from_video(gesture_location, gesture_file_name, mid_frame_counter):
    """
    Identifies the gesture in a video by comparing its features with the training data.

    Args:
        gesture_location (str): Directory containing the video file.
        gesture_file_name (str): Name of the video file.
        mid_frame_counter (int): Frame number to extract.

    Returns:
        Bunch: Details of the identified gesture.
    """
    video_feature = extract_image_feature(gesture_location, gesture_file_name, mid_frame_counter)

    # Initialize default gesture detail
    gesture_detail = Bunch(gesture_key="", gesture_name="", output_label="")

    # Find the closest match using cosine similarity
    cos_sin, gesture_detail = min(
        ((tf.keras.losses.cosine_similarity(video_feature, fv.extracted_feature, axis=-1).numpy(), fv.gesture_detail)
         for fv in featureVectorList),
        key=lambda x: x[0], default=(1.0, gesture_detail)
    )

    print(f"{gesture_file_name} calculated gesture {gesture_detail.gesture_name}")
    return gesture_detail


# Create the gesture details using Bunch instead of a custom class
gesture_details = [
    Bunch(gesture_key="Num0", gesture_name="0", output_label="0"),
    Bunch(gesture_key="Num1", gesture_name="1", output_label="1"),
    Bunch(gesture_key="Num2", gesture_name="2", output_label="2"),
    Bunch(gesture_key="Num3", gesture_name="3", output_label="3"),
    Bunch(gesture_key="Num4", gesture_name="4", output_label="4"),
    Bunch(gesture_key="Num5", gesture_name="5", output_label="5"),
    Bunch(gesture_key="Num6", gesture_name="6", output_label="6"),
    Bunch(gesture_key="Num7", gesture_name="7", output_label="7"),
    Bunch(gesture_key="Num8", gesture_name="8", output_label="8"),
    Bunch(gesture_key="Num9", gesture_name="9", output_label="9"),
    Bunch(gesture_key="FanDown", gesture_name="Decrease Fan Speed", output_label="10"),
    Bunch(gesture_key="FanOn", gesture_name="FanOn", output_label="11"),
    Bunch(gesture_key="FanOff", gesture_name="FanOff", output_label="12"),
    Bunch(gesture_key="FanUp", gesture_name="Increase Fan Speed", output_label="13"),
    Bunch(gesture_key="LightOff", gesture_name="LightOff", output_label="14"),
    Bunch(gesture_key="LightOn", gesture_name="LightOn", output_label="15"),
    Bunch(gesture_key="SetThermo", gesture_name="SetThermo", output_label="16")
]

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
featureVectorList = []
path_to_train_data = "traindata/"
count = 0
for file in os.listdir(path_to_train_data):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        gesture = find_gesture_by_filename(file)
        if gesture:
            # Instead of GestureFeature, we use Bunch to store both gesture_detail and extracted_feature
            featureVectorList.append(Bunch(gesture_detail=gesture, extracted_feature=extract_image_feature(path_to_train_data, file, count)))
            count += 1

# =============================================================================
# Get the penultimate layer for test data and recognize gestures
# =============================================================================
video_locations = ["test/"]
test_count = 0

with open('Results.csv', 'w', newline='') as results_file:
    csv_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for video_location in video_locations:
        for test_file in os.listdir(video_location):
            if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
                # Recognize the gesture (use cosine similarity for comparing the vectors)
                recognized_gesture_detail = identify_gesture_from_video(video_location, test_file, test_count)
                test_count += 1
                csv_writer.writerow([recognized_gesture_detail.output_label])