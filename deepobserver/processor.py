import argparse
import base64
import logging
import os
from queue import Queue
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

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

        yolo_model: str = "yolov8s.pt"
        logger.info(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.detection_threshold = 0.5
        self.tracked_classes = ['person', 'backpack', 'handbag', 'suitcase',
                              'laptop', 'cell phone', 'knife', 'scissors']
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=40,
            detectShadows=False
        )
        self.min_motion_area = 1000

        # Add FPS monitoring
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        self.frame_queue = Queue(maxsize=2)
        self.running = False

        # Add object categories and their handling rules
        self.object_categories = {
            'security_concern': {
                'objects': [
                    'knife',
                    'scissors',
                    'bottle',  # Could be a weapon or contain harmful substances
                    'fire extinguisher',  # Important for safety monitoring
                    'cell phone',  # Could indicate unauthorized recording
                    'suitcase',  # Unattended baggage
                    'backpack'   # Unattended baggage
                ],
                'alert_level': 'high',
                'min_confidence': 0.6,
                'color': (0, 0, 255),  # Red
                'requires_immediate_alert': True,
                'track_duration': True,  # Track how long object is present
            },
            'valuables': {
                'objects': ['laptop', 'cell phone', 'backpack'],
                'alert_level': 'medium',
                'min_confidence': 0.5,
                'color': (255, 165, 0),  # Orange
            },
            'people': {
                'objects': ['person'],
                'alert_level': 'low',
                'min_confidence': 0.4,
                'color': (0, 255, 0),  # Green
                'track_interactions': True,
            }
        }

        # Add scene analysis
        self.scene_history = {
            'objects': [],  # List of objects over time
            'last_positions': {},  # Track object positions
            'duration_tracking': {},  # Track how long objects present
            'scene_context': None,  # Last scene context
        }

    def analyze_movement_pattern(self, positions: List[Tuple]) -> dict:
        """Analyze movement pattern from position history"""
        total_distance = 0
        for i in range(1, len(positions)):
            p1 = positions[i-1]
            p2 = positions[i]
            center1 = ((p1[0] + p1[2])//2, (p1[1] + p1[3])//2)
            center2 = ((p2[0] + p2[2])//2, (p2[1] + p2[3])//2)
            distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
            total_distance += distance

        return {
            'speed': total_distance / len(positions),
            'direction': 'right' if positions[-1][0] > positions[0][0] else 'left'
        }

    def get_object_category(self, object_name: str) -> Tuple[str, dict]:
        """Get category and rules for an object"""
        for category, rules in self.object_categories.items():
            if object_name in rules['objects']:
                return category, rules
        return 'default', None

    def process_frame_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        results = self.yolo.predict(frame, conf=self.detection_threshold)[0]
        annotated_frame = frame.copy()
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = box.cls[0].item()
            conf = box.conf[0].item()
            class_name = results.names[class_id]

            # Get category and rules for this object
            category, rules = self.get_object_category(class_name)

            # Check if we should track this object
            if rules and conf >= rules['min_confidence']:
                detections.append({
                    'class': class_name,
                    'category': category,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'alert_level': rules['alert_level']
                })

                # Draw box with category-specific color
                color = rules['color'] if rules else (128, 128, 128)
                cv2.rectangle(annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color, 2)

                # Add alert level indicator for high-priority objects
                label = f'{class_name} {conf:.2f}'
                if rules['alert_level'] == 'high':
                    label = '⚠️ ' + label

                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # Add summary to frame
        summary_y = 30
        for category in self.object_categories:
            category_count = len([d for d in detections if d['category'] == category])
            if category_count > 0:
                cv2.putText(annotated_frame,
                          f"{category}: {category_count}",
                          (10, summary_y),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          self.object_categories[category]['color'],
                          2)
                summary_y += 25

        return annotated_frame, detections

    def analyze_scene(self, detections: List[dict], frame: np.ndarray) -> dict:
        """Analyze scene context and temporal patterns"""
        current_time = time.time()
        scene_analysis = {
            'context': {},
            'alerts': [],
            'patterns': [],
            'risk_level': 0
        }

        # Update object tracking
        current_objects = set()
        for det in detections:
            obj_id = f"{det['class']}_{det['box'][0]}_{det['box'][1]}"
            current_objects.add(obj_id)

            # Track duration
            if obj_id not in self.scene_history['duration_tracking']:
                self.scene_history['duration_tracking'][obj_id] = current_time

            # Track position
            if obj_id not in self.scene_history['last_positions']:
                self.scene_history['last_positions'][obj_id] = []
            self.scene_history['last_positions'][obj_id].append(det['box'])

            # Analyze based on category
            if det['alert_level'] == 'high':
                duration = current_time - self.scene_history['duration_tracking'][obj_id]
                scene_analysis['alerts'].append({
                    'type': 'security_concern',
                    'object': det['class'],
                    'duration': duration,
                    'location': det['box']
                })
                scene_analysis['risk_level'] += 2

        # Clean up old tracking data
        for obj_id in list(self.scene_history['duration_tracking'].keys()):
            if obj_id not in current_objects:
                del self.scene_history['duration_tracking'][obj_id]
                if obj_id in self.scene_history['last_positions']:
                    del self.scene_history['last_positions'][obj_id]

        # Analyze movement patterns
        for obj_id, positions in self.scene_history['last_positions'].items():
            if len(positions) > 5:  # Need enough positions to analyze
                movement = self.analyze_movement_pattern(positions[-5:])
                if movement['speed'] > 50:  # Pixels per frame
                    scene_analysis['patterns'].append({
                        'object': obj_id,
                        'type': 'rapid_movement',
                        'details': movement
                    })

        return scene_analysis

    def process_stream(self) -> None:
        """Process the video stream continuously"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        # Enhanced camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Verify settings
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera settings - FPS: {actual_fps}, Resolution: {actual_width}x{actual_height}")
        last_process_time: float = 0
        frames_buffer: list[np.ndarray] = []

        while(self.cap.isOpened()):
            ret: bool
            frame: np.ndarray
            ret, frame = self.cap.read()
            annotated_frame, detections = self.process_frame_yolo(frame)
            scene_analysis = self.analyze_scene(detections, frame)

            # Add scene analysis visualization
            h, w = frame.shape[:2]

            # Add risk level
            if scene_analysis['risk_level'] > 0:
                risk_color = (0, 0, 255) if scene_analysis['risk_level'] > 5 else (255, 165, 0)
                cv2.putText(annotated_frame,
                            f"Risk Level: {scene_analysis['risk_level']}",
                            (w-200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            risk_color,
                            2)

            # Add movement patterns
            y_offset = 60
            if scene_analysis['patterns']:
                cv2.putText(annotated_frame,
                            "Movement Patterns:",
                            (w-200, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)
                y_offset += 25
                for pattern in scene_analysis['patterns']:
                    cv2.putText(annotated_frame,
                                f"{pattern['object']}: {pattern['details']['direction']}",
                                (w-200, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1)
                    y_offset += 20

            # Add alerts
            if scene_analysis['alerts']:
                y_offset += 10
                cv2.putText(annotated_frame,
                            "Alerts:",
                            (w-200, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)
                y_offset += 25
                for alert in scene_analysis['alerts']:
                    cv2.putText(annotated_frame,
                                f"{alert['object']}: {alert['duration']:.1f}s",
                                (w-200, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1)
                    y_offset += 20

            if not ret:
                logger.error("Failed to read frame from stream")
                break

            current_time: float = cv2.getTickCount() / cv2.getTickFrequency()
            # Only process frames at specified interval
            if (current_time - last_process_time) >= self.frame_interval:
                frames_buffer.append(annotated_frame)
                cv2.imshow('frame', annotated_frame)
                last_process_time = current_time

                if len(frames_buffer) >= self.clip_length:
                    #self.process_buffer(frames_buffer=frames_buffer)
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