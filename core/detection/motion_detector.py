import cv2

class MotionDetector:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,  # mais tempo para aprender o fundo
            varThreshold=25,  # menos sensível a ruído
            detectShadows=False
        )

        self.frame_count = 0

    def detect(self, frame):
        self.frame_count += 1

        fg_mask = self.background_subtractor.apply(frame)

        # Aguarda o fundo estabilizar
        if self.frame_count < 30:
            return fg_mask * 0  # retorna máscara preta

        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return fg_mask

    def detect_contours(self, motion_mask, min_area=500):
        """
        Detecta contornos a partir da máscara de movimento,
        filtrando pequenos ruídos.
        """

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= min_area:
                valid_contours.append(contour)

        return valid_contours

    def get_bounding_boxes(self, contours):
        """
        Retorna uma lista de bounding boxes (x, y, w, h)
        a partir dos contornos detectados.
        """
        boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        return boxes

    def refine_mask(self, motion_mask):
        """
        Limpa ruídos e fortalece as áreas de movimento
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        cleaned = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2
        )

        cleaned = cv2.dilate(cleaned, kernel, iterations=2)

        return cleaned




