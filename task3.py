import argparse
import glob
import hashlib
import json
import os
import sys
import subprocess
from os.path import exists, basename, join
import shutil
import utils
import models
import numpy as np
from copy import deepcopy as copy
import cv2

MINIMUM_KEYPOINT_CONFIDENT = 0.0
SELECT_GROUP_SCORE_RATIO = 0.2
MAXIMUM_FRAME_DIFFERENCE = 20
MAXIMUM_GROUP_DISTANCE = 5

def load_frame_datas(data_path):
    data_path_list = glob.glob("%s/data_json/frame_*.json" % data_path)
    frame_data_list = []
    number_of_peoples_in_frames = []
    for i in range(len(data_path_list)):
        file_path = join(data_path, 'data_json', 'frame_%d.json' % i)
        with open(file_path, 'r') as f:
            pose_result = json.loads(f.read())
            peoples = map(lambda people: utils.split_list(people['pose_keypoints'], 3), pose_result['people'])
            number_of_peoples_in_frames.append(len(peoples))
            frame_data_list.append(peoples)

    return frame_data_list, number_of_peoples_in_frames


def dissimilarity(keypoints1, keypoints2):
    joint_found1 = [
            index for (index, keypoint) in enumerate(keypoints1) \
            if keypoint[2] > MINIMUM_KEYPOINT_CONFIDENT
        ]

    joint_found2 = [
            index for (index, keypoint) in enumerate(keypoints2) \
            if keypoint[2] > MINIMUM_KEYPOINT_CONFIDENT
        ]

    joint_overlap = np.intersect1d(joint_found1, joint_found2)

    joint_position1 = np.array([keypoints1[i][0:2] for i in joint_overlap])
    joint_position2 = np.array([keypoints2[i][0:2] for i in joint_overlap])
    
    try:
        distance = np.mean(np.sqrt(np.sum(((joint_position1 - joint_position2)**2), axis=1)))
    except:
        distance = np.nan

    return distance, joint_overlap


def frame_dissimilarity(peoples1, peoples2):
    peoples_in_current_frame = peoples1
    peoples_in_next_frame = peoples2

    number_of_people_in_current_frame = len(peoples_in_current_frame)
    number_of_people_in_next_frame = len(peoples_in_next_frame)

    joint_distances = \
        np.zeros(shape=(number_of_people_in_current_frame, number_of_people_in_next_frame))
    joint_overlaps = \
        np.zeros(shape=(number_of_people_in_current_frame, number_of_people_in_next_frame))

    min_joint_distance = np.inf

    for people_index1 in range(number_of_people_in_current_frame):
        for people_index2 in range(number_of_people_in_next_frame):
            keypoints1 = peoples_in_current_frame[people_index1]
            keypoints2 = peoples_in_next_frame[people_index2]

            distance, joint_overlap = dissimilarity(keypoints1, keypoints2)

            joint_overlaps[people_index1][people_index2] = len(joint_overlap)

            joint_distances[people_index1][people_index2] = distance
            min_joint_distance = min(min_joint_distance, distance)

    return joint_distances, joint_overlaps, min_joint_distance


def dissimilarity_matrix(frame_data_list):
    frame_data_size = len(frame_data_list)
    joint_distance_matrix = [None] * frame_data_size
    joint_overlap_matrix = [None] * frame_data_size
    min_joint_distance = np.inf

    for keyframe in range(frame_data_size - 1):
        current_keyframe = keyframe
        next_keyframe = current_keyframe + 1

        peoples_in_current_frame = frame_data_list[current_keyframe]
        peoples_in_next_frame = frame_data_list[next_keyframe]

        joint_distances, joint_overlaps, joint_distance = \
            frame_dissimilarity(peoples_in_current_frame, peoples_in_next_frame)

        joint_distance_matrix[keyframe] = joint_distances
        joint_overlap_matrix[keyframe] = joint_overlaps
        min_joint_distance = min(min_joint_distance, joint_distance)

    return joint_distance_matrix, joint_overlap_matrix, min_joint_distance


def match_people_between_frame(frame_data_list, joint_distance_matrix):
    frame_data_size = len(frame_data_list)

    match_pair_matrix = [None] * frame_data_size
    label_matrix = np.full((max(number_of_peoples_in_frames), frame_data_size), np.nan)
    latest_label = number_of_peoples_in_frames[0]

    label_matrix[:number_of_peoples_in_frames[0],0] = np.arange(number_of_peoples_in_frames[0])

    for keyframe in range(frame_data_size-1):
        match_pairs = []
        matrix = copy(joint_distance_matrix[keyframe])

        have_unmatch_pair = lambda matrix: not np.all(np.isnan(matrix))
        while have_unmatch_pair(matrix):
            min_distance = np.nanmin(matrix)
            row, column = np.argwhere(matrix == min_distance)[0]

            match_pairs.append([row, column, min_distance])
            label_matrix[column, keyframe + 1] = label_matrix[row, keyframe]

            matrix[row,:] = np.nan
            matrix[:, column] = np.nan
        
        number_of_people_in_next_frame = number_of_peoples_in_frames[keyframe + 1]
        if len(match_pairs):
            unlabel_peoples = np.setdiff1d(
                    np.arange(number_of_people_in_next_frame), 
                    np.array(match_pairs)[:,1]
                )
        else:
            unlabel_peoples = np.arange(number_of_people_in_next_frame)

        label_matrix[unlabel_peoples, keyframe + 1] = latest_label + np.arange(len(unlabel_peoples))
        latest_label += len(unlabel_peoples)
        match_pair_matrix[keyframe] = match_pairs

    return label_matrix


def group_labels(frame_data_list, label_matrix):
    number_of_group = np.int_(np.nanmax(label_matrix))
    frame_group_starts = np.zeros(number_of_group)
    frame_group_ends = np.zeros(number_of_group)
    frame_group_confidences = np.zeros(number_of_group)
    max_confidence_frame_ids = np.zeros(number_of_group)

    for group_id in range(number_of_group):
        match_locations =  np.argwhere(label_matrix == group_id)
        match_peoples = match_locations[:, 0]
        match_keyframes = match_locations[:, 1]
        
        frame_group_starts[group_id] = np.min(match_keyframes)
        frame_group_ends[group_id] = np.max(match_keyframes)
        confidence = []

        for i in range(len(match_locations)):
            keypoint = np.array(frame_data_list[match_keyframes[i]][match_peoples[i]])
            confidence.append(np.mean(keypoint[:,2]))

        frame_group_confidences[group_id] = np.mean(confidence)
        max_confidence_keyframe_id = np.argmax(confidence)
        max_confidence_frame_ids[group_id] = match_keyframes[max_confidence_keyframe_id]
    
    frame_group_length = frame_group_ends - frame_group_starts
    frame_group_score = frame_group_length * frame_group_confidences

    return number_of_group, frame_group_starts, frame_group_ends, \
        frame_group_length, frame_group_score, max_confidence_frame_ids


def match_frame_group(frame_data_list, label_matrix, frame_group_score):
    selected_group_ids = \
        np.argwhere(frame_group_score > SELECT_GROUP_SCORE_RATIO * np.max(frame_group_score))[:,0]

    match_group_pairs = []
    for group_id1 in selected_group_ids:
        for group_id2 in selected_group_ids:
            if group_id1 == group_id2:
                continue
            
            frame_different = frame_group_starts[group_id2] - frame_group_ends[group_id1]

            if frame_different > 0 and frame_different < MAXIMUM_FRAME_DIFFERENCE:
                keyframe1 = int(frame_group_ends[group_id1])
                people_index1 = np.argwhere(label_matrix[:, keyframe1] == group_id1)[0,0]
                keypoint1 = frame_data_list[keyframe1][people_index1]

                keyframe2 = int(frame_group_starts[group_id2])
                people_index2 = np.argwhere(label_matrix[:, keyframe2] == group_id2)[0,0]
                keypoint2 = frame_data_list[keyframe2][people_index2]

                distance, joint_overlap = dissimilarity(keypoint1, keypoint2)

                normalize_distance = distance / frame_different

                if normalize_distance < MAXIMUM_GROUP_DISTANCE:
                    match_group_pairs.append([group_id1, group_id2, normalize_distance])
                
            elif frame_different <= 0 and frame_different > -MAXIMUM_FRAME_DIFFERENCE:
                keyframe1 = int(frame_group_starts[group_id2] - 1)
                people_index1 = np.argwhere(label_matrix[:, keyframe1] == group_id1)[0,0]
                keypoint1 = frame_data_list[keyframe1][people_index1]

                keyframe2 = int(frame_group_ends[group_id1] + 1)
                people_index2 = np.argwhere(label_matrix[:, keyframe2] == group_id2)[0,0]
                keypoint2 = frame_data_list[keyframe2][people_index2]

                distance, joint_overlap = dissimilarity(keypoint1, keypoint2)

                normalize_distance = distance / ( -frame_different + 2)
                
                if normalize_distance < MAXIMUM_GROUP_DISTANCE:
                    match_group_pairs.append([group_id1, group_id2, normalize_distance])

    return np.array(match_group_pairs), selected_group_ids


def clean_label_matrix(label_matrix, match_group_pairs):
    final_label_matrix = copy(label_matrix)

    for people_index in range(len(final_label_matrix)):
        for keyframe in range(len(final_label_matrix[people_index])):
            label = final_label_matrix[people_index][keyframe]
            if np.isnan(label) or int(label) not in selected_group_ids:
                final_label_matrix[people_index][keyframe] = -1

    if len(match_group_pairs):
        sorted_group_zip = sorted(zip(match_group_pairs[:, 1], range(len(match_group_pairs))), reverse=True)
        sorted_ids = np.array(sorted_group_zip)[:, 1]
        for id in sorted_ids:
            group_id = int(id)
            match_group_id = final_label_matrix == match_group_pairs[group_id, 1]
            final_label_matrix[match_group_id] = match_group_pairs[group_id, 0]

    return final_label_matrix


def copy_result_to_output_path(final_label_matrix, max_confidence_frame_ids, output_path):
    unique_group_id, indices = np.unique(final_label_matrix, return_inverse=True)
    unique_group_id = unique_group_id[1:]
    
    utils.make_directory(output_path)
    count = 1
    for group_id in unique_group_id:
        keyframe = int(max_confidence_frame_ids[int(group_id)])
        people_index = np.argwhere(final_label_matrix[:, keyframe] == int(group_id))[0,0]
        print people_index, keyframe
        finished = False
        while not finished:
            try:
                image_path = join(data_path, 'head', str(keyframe), str(people_index) + '.jpg')
                image_output_path = join(output_path, str(count) + '.jpg')
                image = cv2.imread(image_path)
                cv2.imwrite(image_output_path, image)
                finished = True
            except Exception as e:
                print e
                while keyframe > 0:
                    keyframe -= 1
                    try:
                        people_index = np.argwhere(final_label_matrix[:, keyframe] == int(group_id))[0,0]
                    except:
                        pass

        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data folder")
    parser.add_argument("output_path", help="path to output folder")
    args = parser.parse_args()

    data_path = join(args.data_path)
    output_path = join(args.output_path)

    frame_data_list, number_of_peoples_in_frames = load_frame_datas(data_path)
    frame_data_size = len(frame_data_list)

    joint_distance_matrix, joint_overlap_matrix, min_joint_distance = \
        dissimilarity_matrix(frame_data_list)

    label_matrix = match_people_between_frame(frame_data_list, joint_distance_matrix)

    number_of_group, frame_group_starts, frame_group_ends, frame_group_length \
    , frame_group_score, max_confidence_frame_ids = \
        group_labels(frame_data_list, label_matrix)

    match_group_pairs, selected_group_ids = match_frame_group(frame_data_list, label_matrix, frame_group_score)

    final_label_matrix = clean_label_matrix(label_matrix, match_group_pairs)

    copy_result_to_output_path(final_label_matrix, max_confidence_frame_ids,output_path)
    