import argparse
import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

from prompts import prompts
from llm_client.base import LLMClient, OpenAIClient

logger = logging.getLogger(__name__)
load_dotenv()

class VideoProcessor:
    def __init__(self, rtsp_url: str, fps: float = 1.0) -> None:
        self.rtsp_url: str = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = fps
        self.frame_interval: float = 1.0 / fps # Time interval between frames in second
        self.clip_length: int = 10 # Number of frames to process at once
        self.llm_client: LLMClient = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

    def process_stream(self) -> None:
        """Process the video stream continuously"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        last_process_time: float = 0
        frames_buffer: list[np.ndarray] = []

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
                frames_buffer.append(frame)
                cv2.imshow('frame', frame)
                last_process_time = current_time

                if len(frames_buffer) >= self.clip_length:
                    self.process_buffer(frames_buffer=frames_buffer)
                    # Clear the buffer
                    frames_buffer = []

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Might also need the timestamp of the frames
    def process_buffer(self, frames_buffer: list[np.ndarray]) -> None:
        base64_frames: list[bytes] = []
        print("Processing buffer...")
        for frame in frames_buffer:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame: bytes = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_frame)

        # Step 1: Analyze the scene
        scene_analysis_result: str = self.llm_client.generate_buffer(
            prompt=prompts.CLIP_ANALYSIS_PROMPTS["scene_analysis"],
            base64_images=base64_frames
        )
        print("SCENE ANALYSIS RESULT: ", scene_analysis_result)

        # Step 2: Analyze the temporal changes
        temporal_analysis_result: str = self.llm_client.generate_buffer(
            prompt=prompts.CLIP_ANALYSIS_PROMPTS["temporal_analysis"],
            base64_images=base64_frames
        )
        print("TEMPORAL ANALYSIS RESULT: ", temporal_analysis_result)

        # Step 3: Analyze the semantic meaning of the scene
        semantic_analysis_prompt: str = prompts.CLIP_ANALYSIS_PROMPTS["semantic_analysis"].format(
            scene_analysis=scene_analysis_result,
            temporal_analysis=temporal_analysis_result
        )

        semantic_analysis_result: str = self.llm_client.generate(
            prompt=semantic_analysis_prompt
        )
        print("SEMANTIC ANALYSIS RESULT: ", semantic_analysis_result)
        # TODO: Save the response to vector database

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTSP video stream')
    parser.add_argument('--rtsp-url', type=str, required=True,
                      help='RTSP URL (e.g., rtsp://username:password@ip_address:port/stream)')
    args = parser.parse_args()

    processor = VideoProcessor(args.rtsp_url)
    processor.process_stream()