import cv2
from core.video.video_capture import VideoCapture
from core.processing.frame_processor import FrameProcessor
from core.detection.motion_detector import MotionDetector



def main():
    video = VideoCapture()
    processor = FrameProcessor()
    motion_detector = MotionDetector()


    video.open()

    while True:
        frame = video.read_frame()

        gray_frame = processor.to_gray(frame)
        blurred_frame = processor.apply_blur(gray_frame)
        motion_mask = motion_detector.detect(blurred_frame)
        motion_mask = motion_detector.refine_mask(motion_mask)





        cv2.imshow("Original", frame)
        cv2.imshow("Escala de Cinza", gray_frame)
        cv2.imshow("Cinza com Blur", blurred_frame)
        cv2.imshow("Motion Mask", motion_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
