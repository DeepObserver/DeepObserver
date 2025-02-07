import argparse
import cv2
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        pass

    def process_stream(self):
        """Process the video stream continuously"""
        self.cap = cv2.VideoCapture(self.rtsp_url)

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTSP video stream')
    parser.add_argument('--rtsp-url', type=str, required=True,
                      help='RTSP URL (e.g., rtsp://username:password@ip_address:port/stream)')

    args = parser.parse_args()

    processor = VideoProcessor(args.rtsp_url)
    processor.process_stream()
