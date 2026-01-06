import cv2


class FrameProcessor:
    def __init__(self):
        pass

    def to_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, gray_frame):
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)
