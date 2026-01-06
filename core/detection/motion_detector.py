import cv2

class MotionDetector:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=40,
            detectShadows=True
        )

    def detect(self, frame):
        """
        Recebe um frame em cinza e com blur
        Retorna uma máscara binária do movimento
        """
        fg_mask = self.background_subtractor.apply(frame)

        # Remove sombras (valor 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        return fg_mask

    def detect_contours(self, motion_mask):
        pass

