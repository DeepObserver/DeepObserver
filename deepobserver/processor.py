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
                
                cv2.putText(annotated_frame, 
                          label,
                          (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          color,
                          2)

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

    def process_stream(self) -> None:
        self.running = True
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()

        try:
            while self.running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Run YOLO detection and scene analysis
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
                    
                    # Run LLM analysis if needed
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