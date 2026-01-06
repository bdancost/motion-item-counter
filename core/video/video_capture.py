import cv2


class VideoCapture:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera.")

    def read_frame(self):
        if self.capture is None:
            raise RuntimeError("A câmera não foi aberta.")

        ret, frame = self.capture.read()

        if not ret:
            raise RuntimeError("Não foi possível ler o frame da câmera.")

        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()

