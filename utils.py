import numpy as np
import cv2
import math
import os, errno


def crop_image(image, x, y, height, width):
    image_height, image_width, channels = image.shape 
    if x < 0:
        x = 0
    if x + height >= image_height:
        height = image_height - 1
    if y < 0:
        y = 0
    if y + width >= image_width:
        width = image_width - 1
            
    return image[x:x+height, y:y+width]


def split_list(items, chunk_size):
    return zip(*[iter(items)]*chunk_size)

def color_histogram(image):
    chans = cv2.split(image)
    features = []

    # loop over the image channels
    for chan in chans:
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

    # here we are simply showing the dimensionality of the
    # flattened color histogram 256 bins for each channel
    # x 3 channels = 768 total values -- in practice, we would
    # normally not use 256 bins for each channel, a choice
    # between 32-96 bins are normally used, but this tends
    # to be application dependent
    # print "flattened feature vector size: %d" % (np.array(features).flatten().shape)

    return np.array(features).flatten()

# https://stackoverflow.com/questions/42584259/python-code-to-calculate-angle-between-three-points-lat-long-coordinates/42616370#42616370
def angle_between_vectors_degrees(u, v):
    """Return the angle between two vectors in any dimension space,
    in degrees."""
    degree = np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    return degree if degree <= 180 else 360 - degree

def make_directory(name):
    import os, errno
    try:
        os.makedirs(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


if '__main__' == __name__:
    print split_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
