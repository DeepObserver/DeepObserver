import argparse
import base64
import logging
import os
from typing import Optional, Tuple, List
import time
import threading
from queue import Queue

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from llm_client.base import LLMClient, OpenAIClient, ClaudeClient

logger = logging.getLogger(__name__)
load_dotenv()

class VideoProcessor:
    def __init__(self, rtsp_url: str, fps: float = 60.0, yolo_model: str = 'yolov8s.pt') -> None:
        self.rtsp_url: str = rtsp_url
        self.fps: float = fps
        self.frame_interval: float = 1.0 / fps
        
        # Initialize LLM client
        self.llm_clients = [
            OpenAIClient(api_key=os.getenv("OPENAI_API_KEY")),
        ]
        
        # Initialize YOLO model
        logger.info(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.detection_threshold = 0.5
        self.tracked_classes = ['person', 'backpack', 'handbag', 'suitcase', 
                              'laptop', 'cell phone', 'knife', 'scissors']
        
        # Initialize motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=40,
            detectShadows=False
        )
        self.min_motion_area = 1000
        self.last_significant_change = time.time()
        
        # Add FPS monitoring
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        self.frame_queue = Queue(maxsize=2)
        self.running = False

    def process_frame_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        results = self.yolo.predict(frame, conf=self.detection_threshold)[0]
        annotated_frame = frame.copy()
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = box.cls[0].item()
            conf = box.conf[0].item()
            class_name = results.names[class_id]
            
            if class_name in self.tracked_classes:
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
                
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                cv2.putText(annotated_frame, 
                          f'{class_name} {conf:.2f}',
                          (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (0, 255, 0),
                          2)

        return annotated_frame, detections

    def process_stream(self) -> None:
        self.running = True
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()

        try:
            while self.running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Run YOLO detection
                    annotated_frame, detections = self.process_frame_yolo(frame)
                    
                    if self.should_analyze_frame(frame):
                        _, buffer = cv2.imencode('.jpg', frame)
                        base64_frame: bytes = base64.b64encode(buffer).decode('utf-8')
                        
                        has_changes = False
                        analyses = []
                        
                        for llm_client in self.llm_clients:
                            response = llm_client.generate(
                                prompt="Analyze this camera feed", 
                                base64_image=base64_frame
                            )
                            if response != "situation unchanged":
                                has_changes = True
                                analyses.append((llm_client.name, response))
                        
                        if has_changes:
                            print("\n" + "-"*50)
                            for name, analysis in analyses:
                                print(f"{name} Analysis:")
                                print(analysis.strip())
                            print("-"*50)

                    cv2.imshow('frame', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

        finally:
            self.running = False
            capture_thread.join(timeout=1)
            cv2.destroyAllWindows()

    def capture_frames(self):
        """Capture frames in a separate thread"""
        cap = cv2.VideoCapture(self.rtsp_url)
        
        # Enhanced camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Verify settings
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera settings - FPS: {actual_fps}, Resolution: {actual_width}x{actual_height}")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Always get the latest frame
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
                self.frame_queue.put(frame)
                self.fps_counter += 1
            else:
                logger.error("Failed to read frame")
                # Try to reconnect
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)

        cap.release()

    def should_analyze_frame(self, frame: np.ndarray) -> bool:
        """Determine if frame should be analyzed based on motion"""
        # Apply motion detection
        fgmask = self.background_subtractor.apply(frame)
        motion_area = np.sum(fgmask == 255)
        
        # Check if enough time has passed and motion threshold is met
        time_since_last = time.time() - self.last_significant_change
        if motion_area > self.min_motion_area and time_since_last > 2.0:
            self.last_significant_change = time.time()
            return True
            
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTSP video stream')
    parser.add_argument('--rtsp-url', type=str, required=True,
                      help='RTSP URL (e.g., rtsp://username:password@ip_address:port/stream)')
    parser.add_argument('--fps', type=float, default=10.0,
                      help='Frames per second to process (default: 10.0)')
    parser.add_argument('--yolo-model', type=str, default='yolov8s.pt',
                      choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                      help='YOLO model to use (default: yolov8s.pt)')

    args = parser.parse_args()
    processor = VideoProcessor(args.rtsp_url, fps=args.fps, yolo_model=args.yolo_model)
    processor.process_stream()