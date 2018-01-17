import numpy as np


def crop_image(image, x, y, height, width):
    return image[x:x+height, y:y+width]


def split_list(items, chunk_size):
    return zip(*[iter(items)]*chunk_size)


if '__main__' == __name__:
    print split_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
