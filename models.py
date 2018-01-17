import utils


class KeyPoint:
    def __init__(self, keypoint_vector):
        self.key_points = utils.split_list(keypoint_vector, 3)

    def box(self):
        xs = [int(key[0]) for key in self.key_points]
        ys = [int(key[1]) for key in self.key_points]

        return min(xs), max(xs), min(ys), max(ys)

    def nose(self):
        return self.key_points[0]

    def neck(self):
        return self.key_points[1]

    def right_shoulder(self):
        return self.key_points[2]

    def right_elbow(self):
        return self.key_points[3]

    def right_wrist(self):
        return self.key_points[4]

    def left_shoulder(self):
        return self.key_points[5]

    def left_elbow(self):
        return self.key_points[6]

    def left_wrist(self):
        return self.key_points[7]

    def right_hip(self):
        return self.key_points[8]

    def right_knee(self):
        return self.key_points[9]

    def right_ankle(self):
        return self.key_points[10]

    def left_hip(self):
        return self.key_points[11]

    def left_knee(self):
        return self.key_points[12]

    def left_ankle(self):
        return self.key_points[13]

    def right_eye(self):
        return self.key_points[14]

    def left_eye(self):
        return self.key_points[15]

    def right_ear(self):
        return self.key_points[16]

    def left_ear(self):
        return self.key_points[17]

    def background(self):
        return self.key_points[18]

    def __str__(self):
        return str(self.key_points)


if __name__ == '__main__':
    people = KeyPoint([
        379.596, 69.1232, 0.964077, 380.877, 97.7577, 0.95408, 360.054, 97.7889, 0.933842, 350.917, 125.189, 0.85681,
        367.817, 142.209, 0.892876, 401.718, 96.4875, 0.920628, 409.582, 125.239, 0.901604, 396.529, 144.809, 0.844744,
        370.429, 159.123, 0.86592, 370.511, 202.136, 0.935204, 371.78, 242.622, 0.900671, 393.946, 159.087, 0.802822,
        396.497, 202.117, 0.879038, 395.234, 242.593, 0.876667, 375.673, 66.5721, 0.916479, 382.183, 66.5386, 0.954501,
        370.406, 70.4251, 0.800231, 390.009, 67.9001, 0.929997
    ])

    print people
