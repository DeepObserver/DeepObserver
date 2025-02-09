import argparse
import logging
import os
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

from llm_client.base import LLMClient, OpenAIClient

logger = logging.getLogger(__name__)
load_dotenv()

class VideoProcessor:
    def __init__(self, rtsp_url: str, fps: float = 1.0) -> None:
        self.rtsp_url: str = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = fps
        self.frame_interval: float = 1.0 / fps # Time interval between frames in second
        self.llm_client: LLMClient = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

    def process_stream(self) -> None:
        """Process the video stream continuously"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        last_process_time: float = 0

        while(self.cap.isOpened()):
            ret: bool
            frame: np.ndarray
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from stream")
                break

            current_time: float = cv2.getTickCount() / cv2.getTickFrequency()
            # Only process frames at specified interval
            if (current_time - last_process_time) >= self.frame_interval:
                cv2.imshow('frame', frame)
                last_process_time = current_time
                # This is where you would call your LLM to process the frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTSP video stream')
    parser.add_argument('--rtsp-url', type=str, required=True,
                      help='RTSP URL (e.g., rtsp://username:password@ip_address:port/stream)')

    args = parser.parse_args()

    processor = VideoProcessor(args.rtsp_url)
    processor.process_stream()