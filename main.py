import cv2
from core.video.video_capture import VideoCapture
from core.processing.frame_processor import FrameProcessor


def main():
    video = VideoCapture()
    processor = FrameProcessor()

    video.open()

    while True:
        frame = video.read_frame()

        gray_frame = processor.to_gray(frame)
        blurred_frame = processor.apply_blur(gray_frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Escala de Cinza", gray_frame)
        cv2.imshow("Cinza com Blur", blurred_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
