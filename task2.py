import cv2
import json
import glob
import face_recognition
import models
import utils

vidcap = cv2.VideoCapture("squat.avi")

file_list = glob.glob('squat/*.json')
file_list.sort()

success = True

count = 0; 
while success:
    success,image = vidcap.read()

    if not success:
        break

    print "Process frame %d / %s" % (count, file_list[count])

    utils.make_directory("output/frame")
    cv2.imwrite("output/frame/frame_%d.jpg" % count, image)     # save frame as JPEG file
    
    filename = file_list[count]
    with open(filename, 'r') as f:
        pose_result = json.loads(f.read())
    
    number_of_people = len(pose_result['people'])

    for number in range(number_of_people):
        data = pose_result['people'][number]

        key_point = models.KeyPoint(data['pose_keypoints'])

        # Head
        x_min, y_min, height, width = key_point.head()
        head_image = utils.crop_image(image, x_min, y_min, height, width)

        utils.make_directory("output/head/%d" % count)
        cv2.imwrite("output/head/%d/%d.jpg" % (count, number), head_image)

        # Body
        x_min, y_min, height, width = key_point.box()
        body_image = utils.crop_image(image, x_min, y_min, height, width)
        utils.make_directory("output/body/%d" % count)
        cv2.imwrite("output/body/%d/%d.jpg" % (count, number), body_image)
        
        # Face Features
        face_features = face_recognition.face_encodings(head_image)
        data['face_features'] = face_features

        # bgr histogram
        data['bgr_histogram'] = utils.color_histogram(body_image).tolist()

        # hsv histogram
        body_hsv_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2HSV)
        data['hsv_histogram'] = utils.color_histogram(body_hsv_image).tolist()

        data['left_elbow_angle'] = key_point.left_elbow_angle()
        data['right_elbow_angle'] = key_point.right_elbow_angle()

        data['left_shoulder_angle'] = key_point.left_shoulder_angle()
        data['right_shoulder_angle'] = key_point.right_shoulder_angle()

        data['left_knee_angle'] = key_point.left_knee_angle()
        data['right_knee_angle'] = key_point.right_knee_angle()

        data['left_hip_angle'] = key_point.left_hip_angle()        
        data['right_hip_angle'] = key_point.right_hip_angle()

    utils.make_directory('output/data_json')
    with open("output/data_json/frame_%d.json" % count, 'w') as f:
        f.write(json.dumps(pose_result, indent=4, sort_keys=True))

    count += 1