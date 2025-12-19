"""
Head Pose Detection Module - Image Version
Uses MediaPipe Face Mesh to detect head orientation in static images.
"""

import cv2
import mediapipe as mp
import numpy as np


class HeadPoseDetector:
    """
    Detects head pose in static images using MediaPipe Face Mesh.
    """
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh for head pose detection."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Thresholds for direction detection
        self.horizontal_threshold = 0.03
        self.vertical_threshold = 0.02
        
    def detect_head_pose(self, image):
        """
        Detect head pose from an image.
        
        Args:
            image: Input BGR image
            
        Returns:
            dict: Detection results containing:
                - annotated_image: Image with annotations
                - direction: Head direction (Forward, Left, Right, Up, Down, or combinations)
                - is_suspicious: Boolean indicating if head pose is suspicious
                - confidence: Detection confidence
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        # Create a copy for annotation
        annotated_image = image.copy()
        
        direction = "No Face Detected"
        is_suspicious = True
        confidence = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                img_h, img_w, _ = image.shape
                
                # Get key landmarks
                nose_tip = face_landmarks.landmark[1]
                nose_x = nose_tip.x
                nose_y = nose_tip.y
                
                left_eye = face_landmarks.landmark[263]
                right_eye = face_landmarks.landmark[33]
                chin = face_landmarks.landmark[152]
                forehead = face_landmarks.landmark[10]
                
                # Calculate face center
                face_center_x = (left_eye.x + right_eye.x) / 2
                face_center_y = (forehead.y + chin.y) / 2
                
                # Calculate face dimensions
                face_width = abs(left_eye.x - right_eye.x)
                face_height = abs(chin.y - forehead.y)
                
                # Calculate normalized offset
                horizontal_offset = (nose_x - face_center_x) / face_width if face_width > 0 else 0
                vertical_offset = (nose_y - face_center_y) / face_height if face_height > 0 else 0
                
                # Determine direction
                directions = []
                
                if horizontal_offset < -self.horizontal_threshold:
                    directions.append("Right")
                    is_suspicious = True
                elif horizontal_offset > self.horizontal_threshold:
                    directions.append("Left")
                    is_suspicious = True
                
                if vertical_offset < -self.vertical_threshold:
                    directions.append("Up")
                    is_suspicious = True
                elif vertical_offset > self.vertical_threshold:
                    directions.append("Down")
                    is_suspicious = True
                
                if directions:
                    direction = " + ".join(directions)
                else:
                    direction = "Forward"
                    is_suspicious = False
                
                # Set color based on status
                color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                status = "LOOKING AWAY" if is_suspicious else "LOOKING AT SCREEN"
                
                # Draw visualizations
                nose_px = int(nose_x * img_w)
                nose_py = int(nose_y * img_h)
                cv2.circle(annotated_image, (nose_px, nose_py), 5, color, -1)
                
                center_px = int(face_center_x * img_w)
                center_py = int(face_center_y * img_h)
                cv2.circle(annotated_image, (center_px, center_py), 5, (255, 255, 0), -1)
                
                cv2.line(annotated_image, (center_px, center_py), (nose_px, nose_py), color, 2)
                
                # Add text annotations
                cv2.putText(annotated_image, f"Direction: {direction}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(annotated_image, f"Status: {status}", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                confidence = 0.95
        
        return {
            'annotated_image': annotated_image,
            'direction': direction,
            'is_suspicious': is_suspicious,
            'confidence': confidence
        }
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()
