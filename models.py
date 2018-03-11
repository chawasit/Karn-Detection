import utils
import numpy as np


class KeyPoint:
    def __init__(self, keypoint_vector):
        self.key_points = utils.split_list(keypoint_vector, 3)

    def average_confident(self):
        return np.average(map(lambda x: x[2], self.key_points))

    def box(self, ratio=0.15):
        xs = [int(key[0]) for key in self.key_points if key[2] > 0]
        ys = [int(key[1]) for key in self.key_points if key[2] > 0]

        if len(xs) == 0 or len(ys) == 0:
            return None, None, None, None
            
        x = min(xs)
        y = min(ys)

        height = max(ys) - y
        width = max(xs) - x

        if height < 10 or width < 10:
            return None, None, None, None

        x_offset = max(width * ratio, )
        y_offset = max(height * ratio,)

        return int(y - y_offset), int(x - x_offset) \
               , int(height + y_offset), int(width + x_offset)

    def head(self, ratio=0.5):
        key_points = [self.left_ear(), self.right_ear(), 
                      self.left_eye(), self.right_eye(),
                      self.nose()]

        xs = [int(key[0]) for key in key_points if key[2] > 0]
        ys = [int(key[1]) for key in key_points if key[2] > 0]

        if len(xs) == 0 or len(ys) == 0:
            return None, None, None, None

        x = min(xs)
        y = min(ys)

        width = max(xs) - x 
        height = width * 1.6
        

        if height < 10 or width < 10:
            # x, y, height, width = self.box()
            # if x is None:
            #     return None, None, None, None
            # return x, y, height / 2, width
            return self.box()

        y -= height * 0.35

        x_offset = width * ratio
        y_offset = height * ratio

        return int(y - y_offset), int(x - x_offset) \
               , int(height + 2 * y_offset), int(width + 2 * x_offset)

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

    def _angle(self, a, b, c, confident):
        if a[2] < confident or b[2] < confident or c[2] < confident:
            return None

        ba = np.subtract(a, b)
        bc = np.subtract(c, b)
        
        return utils.angle_between_vectors_degrees(ba[:-1], bc[:-1])

    def left_elbow_angle(self, confident=0.5):
        return self._angle(
            self.left_wrist(),
            self.left_elbow(),
            self.left_shoulder(),
            confident
        )
    
    def right_elbow_angle(self, confident=0.5):
        return self._angle(
            self.right_wrist(),
            self.right_elbow(),
            self.right_shoulder(),
            confident
        )

    def left_shoulder_angle(self, confident=0.5):
        return self._angle(
            self.left_elbow(),
            self.left_shoulder(),
            self.neck(),
            confident
        )
    
    def right_shoulder_angle(self, confident=0.5):
        return self._angle(
            self.right_elbow(),
            self.right_shoulder(),
            self.neck(),
            confident
        )

    def left_knee_angle(self, confident=0.5):
        return self._angle(
            self.left_hip(),
            self.left_knee(),
            self.left_ankle(),
            confident
        )
    
    def right_knee_angle(self, confident=0.5):
        return self._angle(
            self.right_hip(),
            self.right_knee(),
            self.right_ankle(),
            confident
        )

    def left_hip_angle(self, confident=0.5):
        return self._angle(
            self.neck(),
            self.left_hip(),
            self.left_knee(),
            confident
        )
    
    def right_hip_angle(self, confident=0.5):
        return self._angle(
            self.neck(),
            self.right_hip(),
            self.right_knee(),
            confident
        )


if __name__ == '__main__':
    people = KeyPoint([
        379.596, 69.1232, 0.964077, 380.877, 97.7577, 0.95408, 360.054, 97.7889, 0.933842, 350.917, 125.189, 0.85681,
        367.817, 142.209, 0.892876, 401.718, 96.4875, 0.920628, 409.582, 125.239, 0.901604, 396.529, 144.809, 0.844744,
        370.429, 159.123, 0.86592, 370.511, 202.136, 0.935204, 371.78, 242.622, 0.900671, 393.946, 159.087, 0.802822,
        396.497, 202.117, 0.879038, 395.234, 242.593, 0.876667, 375.673, 66.5721, 0.916479, 382.183, 66.5386, 0.954501,
        370.406, 70.4251, 0.800231, 390.009, 67.9001, 0.929997
    ])

    print people
