import cv2
import json
import glob
# import face_recognition
import models
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video", help="path to video file")
parser.add_argument("data", help="path wildcard to openpose json result")
parser.add_argument("output", help="output path")
parser.add_argument("--frame", help="save extract frames from video", action="store_true")
args = parser.parse_args()

video_path = args.video
data_path = args.data
output_path = args.output

print "video path: %s" % video_path
print "data path: %s" % data_path
print "output path: %s" % output_path

vidcap = cv2.VideoCapture(video_path)

file_list = glob.glob("%s/*.json" % data_path)
file_list.sort()

success = True

count = 0; 
while success:
    success,image = vidcap.read()

    if not success:
        break

    print "Process frame %d / %s" % (count, file_list[count])

    if args.frame:
        utils.make_directory("%s/frame" % output_path)
        cv2.imwrite("%s/frame/frame_%d.jpg" % (output_path, count), image)     # save frame as JPEG file

    height, width, channels = image.shape

    filename = file_list[count]
    with open(filename, 'r') as f:
        pose_result = json.loads(f.read())
    
    number_of_people = len(pose_result['people'])

    for number in range(number_of_people):
        data = pose_result['people'][number]

        key_point = models.KeyPoint(data['pose_keypoints'])

        # Head
        x, y, height, width = key_point.head(image_height=height, image_width=width)
        face_features = []
        data['head'] = False
        if x and y:
            head_image = utils.crop_image(image, x, y, height, width)
            utils.make_directory("%s/head/%d" % (output_path, count))
            cv2.imwrite("%s/head/%d/%d.jpg" % (output_path, count, number), head_image)

            # try:
            #     face_features = face_recognition.face_encodings(head_image)[0].tolist()
            # except Exception as e:
            #     pass

            data['head'] = True

        # Face Features
        data['face_features'] = face_features

        # Body
        x, y, height, width = key_point.box(image_height=height, image_width=width)
        bgr_histogram = hsv_histogram = []
        data['body'] = False
        if x and y:
            body_image = utils.crop_image(image, x, y, height, width)
            utils.make_directory("%s/body/%d" % (output_path, count))
            cv2.imwrite("%s/body/%d/%d.jpg" % (output_path, count, number), body_image)

            bgr_histogram = utils.color_histogram(body_image).tolist()

            body_hsv_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2HSV)
            hsv_histogram = utils.color_histogram(body_hsv_image).tolist()
            data['body'] = True
        
        data['bgr_histogram'] = bgr_histogram
        data['hsv_histogram'] = hsv_histogram

        data['left_elbow_angle'] = key_point.left_elbow_angle()
        data['right_elbow_angle'] = key_point.right_elbow_angle()

        data['left_shoulder_angle'] = key_point.left_shoulder_angle()
        data['right_shoulder_angle'] = key_point.right_shoulder_angle()

        data['left_knee_angle'] = key_point.left_knee_angle()
        data['right_knee_angle'] = key_point.right_knee_angle()

        data['left_hip_angle'] = key_point.left_hip_angle()        
        data['right_hip_angle'] = key_point.right_hip_angle()

        data['average_confident'] = key_point.average_confident()

        data['body_bounding_box'] = key_point.box()
        data['head_bounding_box'] = key_point.head()

    utils.make_directory('%s/data_json' % output_path)
    with open("%s/data_json/frame_%d.json" % (output_path, count), 'w') as f:
        f.write(json.dumps(pose_result, indent=4, sort_keys=True))

    count += 1