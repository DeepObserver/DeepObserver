import cv2
import numpy as np
from typing import List, Tuple

class QueryInterface:
    def __init__(self, window_width: int = 800, window_height: int = 600):
        self.window_width = window_width
        self.window_height = window_height
        self.window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Window regions
        self.header_height = 100
        self.input_height = 80
        self.content_height = window_height - self.header_height - self.input_height
        
        # Adjust text settings for better wrapping
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.line_spacing = 25
        self.text_margin = 60  # Combined left and right margins
        self.max_line_width = self.window_width - self.text_margin
        
        # State
        self.current_query = ""
        self.responses = []
        self.scroll_position = 0

    def get_text_size(self, text: str) -> Tuple[int, int]:
        """Get actual pixel width of text in the window"""
        (width, height), _ = cv2.getTextSize(text, self.font, self.font_scale, 1)
        return width, height

    def wrap_text(self, text: str, max_width: int, start_x: int = 40) -> List[str]:
        """Wrap text to fit window width"""
        # Calculate actual available width
        available_width = self.window_width - start_x - 40  # Account for margins
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Test width with added word
            test_line = current_line + (" " if current_line else "") + word
            width, _ = self.get_text_size(test_line)
            
            if width <= available_width:
                # Word fits, add it
                current_line = test_line
            else:
                # Line is full, start new line
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        # Add the last line
        if current_line:
            lines.append(current_line)
            
        return lines

    def draw_text_block(self, text: str, start_x: int, start_y: int, color: Tuple[int, int, int]) -> int:
        """Draw text block with proper wrapping"""
        lines = self.wrap_text(text, self.window_width, start_x)
        y_pos = start_y
        
        for line in lines:
            if self.header_height < y_pos < (self.window_height - self.input_height):
                cv2.putText(
                    self.window,
                    line,
                    (start_x, y_pos),
                    self.font,
                    self.font_scale,
                    color,
                    1,
                    cv2.LINE_AA
                )
            y_pos += self.line_spacing
        
        return y_pos

    def update_window(self):
        """Update the query window display"""
        self.window.fill(0)
        
        # Draw header
        cv2.rectangle(self.window, (0, 0), 
                     (self.window_width, self.header_height), (40, 40, 40), -1)
        cv2.putText(self.window, "DeepObserver Query Interface",
                   (20, 40), self.font, 1.0, (255, 255, 255), 2)
        cv2.putText(self.window, "Type question & press Enter | ESC to clear | [/] to scroll",
                   (20, 80), self.font, 0.5, (200, 200, 200), 1)
        
        # Draw input area
        input_y = self.window_height - self.input_height
        cv2.rectangle(self.window, (0, input_y),
                     (self.window_width, self.window_height), (40, 40, 40), -1)
        cv2.putText(self.window, f"> {self.current_query}_",
                   (20, input_y + 40), self.font, 0.8, (0, 255, 0), 2)
        
        # Draw responses with improved wrapping
        y_pos = self.header_height + 20 - self.scroll_position
        for resp in self.responses:
            if y_pos > self.header_height:
                # Question
                question_text = f"Q: {resp['query']}"
                y_pos = self.draw_text_block(
                    question_text,
                    20,  # Left margin for questions
                    y_pos,
                    (200, 200, 200)
                )
                y_pos += 10

                # Answer
                y_pos = self.draw_text_block(
                    resp['response'],
                    40,  # Increased indent for answers
                    y_pos,
                    (0, 255, 0) if resp['response'] != 'Processing...' else (0, 165, 255)
                )

                # Separator
                if y_pos < (self.window_height - self.input_height):
                    cv2.line(
                        self.window,
                        (20, y_pos + 5),
                        (self.window_width - 20, y_pos + 5),
                        (70, 70, 70),
                        1
                    )
                y_pos += 30

        # Draw scroll indicators
        if self.scroll_position > 0:
            cv2.putText(self.window, "▲ More Above",
                       (350, self.header_height + 15),
                       self.font, 0.5, (150, 150, 150), 1)
        if y_pos > (self.window_height - self.input_height):
            cv2.putText(self.window, "▼ More Below",
                       (350, self.window_height - self.input_height - 15),
                       self.font, 0.5, (150, 150, 150), 1)

        cv2.imshow('Query Interface', self.window) 