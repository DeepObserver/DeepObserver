import argparse
import base64
import logging
import os
from queue import Queue, Full
import time
from typing import List, Optional, Tuple
import threading

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from prompts import CLIP_ANALYSIS_PROMPTS
from llm_client.base import LLMClient, OllamaClient, OpenAIClient

logger = logging.getLogger(__name__)
load_dotenv()

class VideoProcessor:
    def __init__(self, rtsp_url: str, fps: float = 60.0, yolo_model: str = 'yolov8s.pt',
                 llm_backend: str = 'openai') -> None:
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
        self.running = True

        # Expand categories beyond security focus
        self.object_categories = {
            'human_activity': {
                'objects': ['person'],
                'track_interactions': True,
                'track_gestures': True,
                'track_groups': True,
                'analyze_behavior': True
            },
            'objects_of_interest': {
                'objects': [
                    'laptop', 'phone', 'book', 'chair', 'table',
                    'cup', 'bottle', 'bag', 'clothing'
                ],
                'track_usage': True,
                'track_placement': True
            },
            'environment': {
                'track_lighting': True,
                'track_occupancy': True,
                'track_space_usage': True
            },
            'interactions': {
                'human_object': True,
                'human_human': True,
                'object_object': True
            }
        }

        # Add contextual understanding
        self.scene_context = {
            'time_of_day': None,
            'activity_level': 0,
            'space_type': None,  # office, home, public, etc.
            'recurring_patterns': {},
            'normal_state': None
        }

        # Initialize LLM client based on backend choice // added backend ollama to blend with LLaVA
        if llm_backend == 'openai':
            self.llm_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        elif llm_backend == 'ollama':
            self.llm_client = OllamaClient(model_name="llava")
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_backend}")

        # Add batch processing queue
        self.batch_queue = Queue(maxsize=5)  # Store frame batches for LLaVA
        self.batch_processing = False
        self.batch_thread = None

        # Separate queues for YOLO and LLaVA
        self.yolo_queue = Queue(maxsize=2)  # Real-time YOLO processing
        self.llava_queue = Queue(maxsize=30)  # Deeper scene analysis
        self.analysis_interval = 5.0  # Seconds between deep analyses

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
        # Map common YOLO classes to our categories
        object_category_map = {
            'person': 'human_activity',
            'laptop': 'objects_of_interest',
            'phone': 'objects_of_interest',
            'book': 'objects_of_interest',
            'chair': 'objects_of_interest',
            'table': 'objects_of_interest',
            'cup': 'objects_of_interest',
            'bottle': 'objects_of_interest',
            'bag': 'objects_of_interest',
            'backpack': 'objects_of_interest',
            'handbag': 'objects_of_interest',
            'suitcase': 'objects_of_interest',
            'cell phone': 'objects_of_interest',
            'couch': 'objects_of_interest',
            'tv': 'objects_of_interest'
        }
        
        # Get category from map, default to 'environment' if not found
        category = object_category_map.get(object_name, 'environment')
        
        # Get the rules for this category
        rules = self.object_categories.get(category, {})
        
        return category, rules

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

            # Add detection if confidence meets threshold
            if conf >= self.detection_threshold:
                detections.append({
                    'class': class_name,
                    'category': category,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })

                # Draw box with category color
                color = (0, 255, 0)  # Default green
                if category == 'human_activity':
                    color = (255, 0, 0)  # Blue
                elif category == 'objects_of_interest':
                    color = (0, 165, 255)  # Orange

                cv2.rectangle(annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color, 2)

                # Add label
                label = f'{class_name} {conf:.2f}'
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
                          (255, 255, 255),  # White text for summary
                          2)
                summary_y += 25

        return annotated_frame, detections

    def analyze_scene(self, detections: List[dict], frame: np.ndarray) -> dict:
        """Enhanced scene analysis"""
        scene_analysis = {
            'spatial_analysis': self.analyze_spatial_relationships(detections),
            'activity_analysis': self.analyze_activities(detections),
            'context_analysis': self.analyze_context(frame),
            'interaction_analysis': self.analyze_interactions(detections),
            'environmental_analysis': self.analyze_environment(frame)
        }
        
        # Update scene history with new insights
        self.update_scene_history(scene_analysis)
        
        return scene_analysis

    def analyze_spatial_relationships(self, detections: List[dict]) -> dict:
        """Analyze how objects are positioned relative to each other"""
        relationships = {}
        for det1 in detections:
            for det2 in detections:
                if det1 != det2:
                    rel = self.calculate_spatial_relationship(det1['box'], det2['box'])
                    relationships[f"{det1['class']}_{det2['class']}"] = rel
        return relationships

    def analyze_activities(self, detections: List[dict]) -> dict:
        """Analyze ongoing activities in the scene"""
        activities = {
            'individual_actions': [],
            'group_activities': [],
            'object_interactions': []
        }
        # Implementation here
        return activities

    def start_batch_processing(self):
        """Start the background thread for LLaVA processing"""
        self.batch_processing = True
        self.batch_thread = threading.Thread(target=self.process_batches, daemon=True)
        self.batch_thread.start()

    def process_batches(self):
        """Background thread to process batches with LLaVA"""
        while self.batch_processing:
            if not self.batch_queue.empty():
                frames_batch = self.batch_queue.get()
                self.process_buffer(frames_batch)
            else:
                time.sleep(0.1)  # Prevent CPU thrashing

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

        # Create window in main thread
        cv2.namedWindow('YOLO Detections', cv2.WINDOW_NORMAL)

        # Start LLaVA processing thread
        llava_thread = threading.Thread(target=self.llava_processing_loop, daemon=True)
        llava_thread.start()

        frames_buffer = []
        last_analysis_time = time.time()

        try:
            while(self.cap.isOpened() and self.running):
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process YOLO in main thread
                annotated_frame, detections = self.process_frame_yolo(frame)
                cv2.imshow('YOLO Detections', annotated_frame)

                # Collect frames for LLaVA analysis
                current_time = time.time()
                if current_time - last_analysis_time >= self.analysis_interval:
                    frames_buffer.append(frame.copy())
                    
                    if len(frames_buffer) >= self.clip_length:
                        try:
                            self.llava_queue.put(frames_buffer.copy(), block=False)
                            frames_buffer = []
                            last_analysis_time = current_time
                        except Full:
                            logger.warning("LLaVA queue full, skipping analysis")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def llava_processing_loop(self):
        """Deep scene analysis loop"""
        while self.running:
            if not self.llava_queue.empty():
                frames = self.llava_queue.get()
                try:
                    self.process_buffer(frames)
                except Exception as e:
                    logger.error(f"Error in LLaVA processing: {e}")
            time.sleep(0.1)  # Prevent CPU thrashing

    def process_buffer(self, frames_buffer: list[np.ndarray]) -> None:
        base64_frames: list[bytes] = []
        yolo_detections: list[dict] = []
        timestamp = time.strftime("%H:%M:%S")
        logger.info(f"Processing buffer at {timestamp}...")

        # Process frames with both YOLO and prepare for LLaVA
        for frame in frames_buffer:
            # Get YOLO detections
            _, detections = self.process_frame_yolo(frame)
            yolo_detections.append(detections)
            
            # Prepare frame for LLaVA
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame: bytes = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_frame)

        # Summarize YOLO detections
        yolo_summary = self.summarize_detections(yolo_detections)

        # Step 1: Analyze the scene with YOLO context
        scene_prompt = f"""[{timestamp}] 
YOLO Detections: {yolo_summary}

{CLIP_ANALYSIS_PROMPTS['scene_analysis']}"""

        scene_analysis_result: str = self.llm_client.generate_buffer(
            prompt=scene_prompt,
            base64_images=base64_frames
        )
        logger.info(f"Scene Analysis: {scene_analysis_result}")

        # Step 2: Analyze temporal changes with detection context
        temporal_prompt = f"""[{timestamp}]
Previous Detections: {yolo_summary}

{CLIP_ANALYSIS_PROMPTS['temporal_analysis']}"""

        temporal_analysis_result: str = self.llm_client.generate_buffer(
            prompt=temporal_prompt,
            base64_images=base64_frames
        )
        logger.info(f"Temporal Analysis: {temporal_analysis_result}")

        # Step 3: Analyze the semantic meaning
        semantic_analysis_prompt: str = f"[{timestamp}] " + CLIP_ANALYSIS_PROMPTS["semantic_analysis"].format(
            scene_analysis=scene_analysis_result,
            temporal_analysis=temporal_analysis_result
        )

        semantic_analysis_result: str = self.llm_client.generate(
            prompt=semantic_analysis_prompt
        )
        logger.info(f"Semantic Analysis: {semantic_analysis_result}")
        # TODO: Save the response to vector database

    def summarize_detections(self, detections_list: list[dict]) -> str:
        """Create a human-readable summary of YOLO detections"""
        summary = []
        
        # Collect all unique objects and their counts
        object_counts = {}
        for frame_detections in detections_list:
            for det in frame_detections:
                obj = det['class']
                conf = det['confidence']
                if obj not in object_counts:
                    object_counts[obj] = {'count': 0, 'avg_conf': 0}
                object_counts[obj]['count'] += 1
                object_counts[obj]['avg_conf'] += conf

        # Calculate averages and format summary
        for obj, data in object_counts.items():
            avg_conf = data['avg_conf'] / data['count']
            summary.append(f"{obj}: {data['count']} instances (avg conf: {avg_conf:.2f})")

        return "\n".join([
            "YOLO detected the following objects:",
            *summary,
            "\nPlease consider these detections in your analysis."
        ])

    def update_scene_history(self, new_analysis: dict) -> None:
        """Learn from new observations"""
        # Update normal state model
        if self.scene_context['normal_state'] is None:
            self.scene_context['normal_state'] = new_analysis
        else:
            self.scene_context['normal_state'] = self.merge_states(
                self.scene_context['normal_state'],
                new_analysis,
                learning_rate=0.1
            )

        # Update recurring patterns
        self.update_patterns(new_analysis)

        # Adjust sensitivity thresholds based on context
        self.adjust_thresholds()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        time.sleep(0.5)  # Give threads time to finish
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTSP video stream')
    parser.add_argument('--rtsp-url', type=str, required=True,
                      help='RTSP URL (e.g., rtsp://username:password@ip_address:port/stream)')
    parser.add_argument('--fps', type=float, default=10.0,
                      help='Frames per second to process (default: 10.0)')
    parser.add_argument('--yolo-model', type=str, default='yolov8s.pt',
                      choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                      help='YOLO model to use (default: yolov8s.pt)')
    parser.add_argument('--llm-backend', type=str, default='openai',
                      choices=['openai', 'ollama'],
                      help='LLM backend to use (default: openai)')

    args = parser.parse_args()
    processor = VideoProcessor(args.rtsp_url, fps=args.fps,
                             yolo_model=args.yolo_model,
                             llm_backend=args.llm_backend)
    processor.process_stream()