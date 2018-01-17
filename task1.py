import utils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import models


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

    return np.array(features).flatten().shape


if __name__ == '__main__':
    # cap = cv2.VideoCapture("C:\Users\User\project\Karn-Detection\Squat1_8_9.avi")
    # if not cap.isOpened():
    #     print "not open"
    image = cv2.imread("frame574.jpg")

    with open("Squat1_8_9_000000000574_keypoints.json", 'r') as f:
        pose_result = json.loads(f.read())

    key_point = models.KeyPoint(pose_result['people'][0]['pose_keypoints'])
    x_min, x_max, y_min, y_max = key_point.box()
    height, width = x_max - x_min, y_max - y_min

    print x_min, y_min, height, width

    cropped = utils.crop_image(image, y_min, x_min, width, height)

    cv2.imshow("cropped", cropped)

    plot_color_histogram(cropped)

    cv2.waitKey()
