import utils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import models
import face_recognition

def plot_color_histogram(image):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # plot the histogram
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    # here we are simply showing the dimensionality of the
    # flattened color histogram 256 bins for each channel
    # x 3 channels = 768 total values -- in practice, we would
    # normally not use 256 bins for each channel, a choice
    # between 32-96 bins are normally used, but this tends
    # to be application dependent
    print "flattened feature vector size: %d" % (np.array(features).flatten().shape)

    plt.show()

    return np.array(features).flatten()


if __name__ == '__main__':
    # cap = cv2.VideoCapture("C:\Users\User\project\Karn-Detection\Squat1_8_9.avi")
    # if not cap.isOpened():
    #     print "not open"
    image = cv2.imread("frame574.jpg")

    with open("squat/Squat1_8_9_000000000574_keypoints.json", 'r') as f:
        pose_result = json.loads(f.read())

    key_point = models.KeyPoint(pose_result['people'][0]['pose_keypoints'])
    # x_min, y_min, height, width = key_point.head(0.1)
    # print x_min, y_min, height, width

    # cropped = utils.crop_image(image, x_min, y_min, height, width)
    # print face_recognition.face_landmarks(cropped)
    # cv2.imshow("cropped", cropped)

    # print plot_color_histogram(cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV))
    # print plot_color_histogram(cropped)

    print utils.angle_between_vectors_degrees()


    cv2.waitKey()
