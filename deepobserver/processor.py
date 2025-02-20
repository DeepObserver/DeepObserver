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

        # Add interactive query queue
        self.query_queue = Queue(maxsize=10)
        self.response_queue = Queue(maxsize=10)
        
        # Create interactive window
        self.create_interactive_window()

        # Create logs directory structure
        os.makedirs('logs', exist_ok=True)
        os.makedirs('logs/qa', exist_ok=True)
        
        # Initialize single session log files
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.observation_log_file = f"logs/ollama_observations_{timestamp}.txt"
        self.qa_log_file = f"logs/qa/qa_session_{timestamp}.txt"
        
        # Initialize observation log
        with open(self.observation_log_file, 'w') as f:
            f.write(f"DeepObserver Scene Analysis Session\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
        
        # Initialize Q&A log
        with open(self.qa_log_file, 'w') as f:
            f.write(f"DeepObserver Q&A Session\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")

    def create_interactive_window(self):
        """Create window for real-time queries using OpenCV"""
        # Create a larger black image for the query window
        self.query_window = np.zeros((600, 800, 3), dtype=np.uint8)  # Increased size
        cv2.namedWindow('Query Interface', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Query Interface', 800, 600)  # Set initial size
        
        # Initialize query state
        self.current_query = ""
        self.responses = []
        self.scroll_position = 0  # Add scroll position tracking
        self.update_query_window()

    def update_query_window(self):
        """Update the query window display"""
        # Create a fresh black background
        self.query_window.fill(0)
        
        # Add header with fixed position
        cv2.rectangle(self.query_window, (0, 0), (800, 100), (40, 40, 40), -1)  # Dark header background
        
        # Add title and instructions
        cv2.putText(self.query_window,
                   "DeepObserver Query Interface",
                   (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.0,
                   (255, 255, 255),
                   2)
        
        cv2.putText(self.query_window,
                   "Type question & press Enter | ESC to clear | PgUp/PgDn to scroll",
                   (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (200, 200, 200),
                   1)
        
        # Add current query input area
        cv2.rectangle(self.query_window, (0, 520), (800, 600), (40, 40, 40), -1)  # Dark input background
        cv2.putText(self.query_window,
                   f"> {self.current_query}_",  # Add cursor
                   (20, 560),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (0, 255, 0),
                   2)

        # Show responses with scrolling
        y_pos = 120 - self.scroll_position
        for resp in self.responses:
            # Draw Q&A box
            if 100 < y_pos < 500:  # Only draw visible items
                # Question
                cv2.putText(self.query_window,
                          f"Q: {resp['query']}",
                          (20, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (200, 200, 200),
                          1)
                y_pos += 30
                
                # Answer with word wrap
                answer = resp['response']
                lines = self.wrap_text(answer, max_width=90)
                for line in lines:
                    if 100 < y_pos < 500:  # Check visibility for each line
                        cv2.putText(self.query_window,
                                  line,
                                  (40, y_pos),  # Indented
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6,
                                  (0, 255, 0) if resp['response'] != 'Processing...' else (0, 165, 255),
                                  1)
                    y_pos += 25
                
                # Add separator
                if y_pos < 500:
                    cv2.line(self.query_window, (20, y_pos+5), (780, y_pos+5), (70, 70, 70), 1)
                y_pos += 40

        # Add scroll indicators if needed
        if len(self.responses) > 0:
            if self.scroll_position > 0:
                cv2.putText(self.query_window, "▲ More Above", (350, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            if y_pos > 500:
                cv2.putText(self.query_window, "▼ More Below", (350, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow('Query Interface', self.query_window)

    def wrap_text(self, text: str, max_width: int) -> list[str]:
        """Wrap text to fit window width"""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = len(word)
            if current_width + word_width + 1 <= max_width:
                current_line.append(word)
                current_width += word_width + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
                
        if current_line:
            lines.append(" ".join(current_line))
            
        return lines

    def log_qa(self, query: str, response: str):
        """Log Q&A to separate file"""
        timestamp = time.strftime("%H:%M:%S")
        with open(self.qa_log_file, 'a') as f:
            f.write(f"\n[{timestamp}]\n")
            f.write(f"Q: {query}\n")
            f.write(f"A: {response}\n")
            f.write("-" * 50 + "\n")

    def process_queries(self):
        """Process real-time queries in background"""
        while self.running:
            if not self.query_queue.empty():
                query_data = self.query_queue.get()
                try:
                    # Prepare frame for LLaVA
                    _, buffer = cv2.imencode('.jpg', query_data['frame'])
                    base64_frame = base64.b64encode(buffer).decode('utf-8')
                    
                    # Create context-aware prompt for real-time Q&A
                    yolo_context = self.summarize_detections([query_data['detections']])
                    qa_prompt = f"""
Current Scene Context:
{yolo_context}

User Question: {query_data['query']}

Please provide a direct and concise answer based on what you see in the current camera view.
Focus specifically on answering the user's question using the visual information available.
"""
                    # Create a separate LLaVA client for Q&A with no logging
                    qa_client = OllamaClient(model_name="llava", disable_logging=True) if isinstance(self.llm_client, OllamaClient) else OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    # Get LLaVA's response using Q&A client
                    response = qa_client.generate_buffer(
                        prompt=qa_prompt,
                        base64_images=[base64_frame]
                    )
                    
                    # Log Q&A to separate file
                    self.log_qa(query_data['query'], response)
                    
                    # Update UI responses
                    for i, r in enumerate(self.responses):
                        if r['query'] == query_data['query'] and r['response'] == 'Processing...':
                            self.responses[i] = {
                                'query': query_data['query'],
                                'response': response
                            }
                            break
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    logger.error(f"Error processing query: {error_msg}")
                    self.log_qa(query_data['query'], f"Error: {error_msg}")
                    # Update UI with error
                    for i, r in enumerate(self.responses):
                        if r['query'] == query_data['query'] and r['response'] == 'Processing...':
                            self.responses[i] = {
                                'query': query_data['query'],
                                'response': error_msg
                            }
                            break
            time.sleep(0.1)

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

        # Start query processing thread
        query_thread = threading.Thread(target=self.process_queries, daemon=True)
        query_thread.start()

        try:
            while(self.cap.isOpened() and self.running):
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Store current frame and detections for queries
                self.current_frame = frame.copy()
                annotated_frame, detections = self.process_frame_yolo(frame)
                self.current_detections = detections

                # Collect frames for LLaVA analysis
                current_time = time.time()
                if current_time - last_analysis_time >= self.analysis_interval:
                    frames_buffer.append(frame.copy())
                    
                    if len(frames_buffer) >= self.clip_length:
                        try:
                            # Queue frames for LLaVA analysis
                            self.llava_queue.put(frames_buffer.copy(), block=False)
                            frames_buffer = []  # Clear buffer after queuing
                            last_analysis_time = current_time
                        except Full:
                            logger.warning("LLaVA queue full, skipping analysis")
                            frames_buffer = []  # Clear buffer if queue is full
                
                # Show frames - need to check if this actually works lol
                cv2.imshow('YOLO Detections', annotated_frame)
                self.update_query_window()

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('\r') or key == ord('\n'):  # Enter
                    self.submit_query()
                elif key == 8 or key == 127:  # Backspace
                    self.current_query = self.current_query[:-1]
                elif key == 27:  # ESC
                    self.current_query = ""
                elif key == ord('['):  # PgUp - scroll up
                    self.scroll_position = max(0, self.scroll_position - 50)
                elif key == ord(']'):  # PgDn - scroll down
                    self.scroll_position = min(max(0, len(self.responses) * 100), self.scroll_position + 50)
                elif 32 <= key <= 126:  # Printable characters
                    self.current_query += chr(key)
                
                self.update_query_window()

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
        timestamp = time.strftime("%H:%M:%S")  # Just time for entries
        logger.info(f"Processing buffer at {timestamp}...")

        # Process frames with both YOLO and prepare for LLaVA
        for frame in frames_buffer:
            _, detections = self.process_frame_yolo(frame)
            yolo_detections.append(detections)
            
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame: bytes = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_frame)

        # Summarize YOLO detections
        yolo_summary = self.summarize_detections(yolo_detections)

        # Append scene analysis to log
        with open(self.observation_log_file, 'a') as f:
            f.write(f"\n[{timestamp}] New Scene Analysis\n")
            f.write("-" * 50 + "\n")
            
            # Scene Analysis
            scene_prompt = f"YOLO Detections: {yolo_summary}\n\n{CLIP_ANALYSIS_PROMPTS['scene_analysis']}"
            scene_analysis_result = self.llm_client.generate_buffer(
                prompt=scene_prompt,
                base64_images=base64_frames
            )
            f.write("\nScene Analysis:\n")
            f.write(scene_analysis_result + "\n")
            
            # Temporal Analysis
            temporal_prompt = f"Previous Detections: {yolo_summary}\n\n{CLIP_ANALYSIS_PROMPTS['temporal_analysis']}"
            temporal_analysis_result = self.llm_client.generate_buffer(
                prompt=temporal_prompt,
                base64_images=base64_frames
            )
            f.write("\nTemporal Analysis:\n")
            f.write(temporal_analysis_result + "\n")
            
            # Semantic Analysis
            semantic_analysis_prompt = CLIP_ANALYSIS_PROMPTS["semantic_analysis"].format(
                scene_analysis=scene_analysis_result,
                temporal_analysis=temporal_analysis_result
            )
            semantic_analysis_result = self.llm_client.generate(
                prompt=semantic_analysis_prompt
            )
            f.write("\nSemantic Analysis:\n")
            f.write(semantic_analysis_result + "\n")
            f.write("-" * 50 + "\n")

        logger.info("Scene analysis completed and logged")

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

    def submit_query(self):
        """Handle query submission"""
        query = self.current_query.strip()
        if query:
            # Get current frame and YOLO detections
            if hasattr(self, 'current_frame') and hasattr(self, 'current_detections'):
                try:
                    # Add to query queue
                    self.query_queue.put({
                        'query': query,
                        'frame': self.current_frame.copy(),
                        'detections': self.current_detections.copy()
                    }, block=False)
                    
                    # Add placeholder response while processing
                    self.responses.append({
                        'query': query,
                        'response': 'Processing...'
                    })
                    
                    # Clear current query
                    self.current_query = ""
                except Full:
                    logger.error("System busy, try again later")

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