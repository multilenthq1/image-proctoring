"""
Eye Tracking Module - Image Version
Detects and analyzes eye movements in static images.
"""

import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    """
    Tracks eye gaze direction in static images using MediaPipe Face Mesh.
    """
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh for eye tracking."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Eye landmark indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris landmark indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Thresholds
        self.gaze_threshold = 0.15
        self.eye_aspect_ratio_threshold = 0.2
        
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)."""
        if len(eye_landmarks) < 6:
            return 1.0
        
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if C == 0:
            return 1.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_iris_position(self, iris_landmarks, eye_landmarks):
        """Calculate iris position relative to eye."""
        if len(iris_landmarks) == 0 or len(eye_landmarks) == 0:
            return 0.5, 0.5
        
        iris_center = np.mean(iris_landmarks, axis=0)
        
        eye_left = np.min(eye_landmarks[:, 0])
        eye_right = np.max(eye_landmarks[:, 0])
        eye_top = np.min(eye_landmarks[:, 1])
        eye_bottom = np.max(eye_landmarks[:, 1])
        
        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top
        
        if eye_width == 0 or eye_height == 0:
            return 0.5, 0.5
        
        horizontal_ratio = (iris_center[0] - eye_left) / eye_width
        vertical_ratio = (iris_center[1] - eye_top) / eye_height
        
        return horizontal_ratio, vertical_ratio
    
    def track_eyes(self, image):
        """
        Track eyes and detect gaze direction in an image.
        
        Args:
            image: Input BGR image
            
        Returns:
            dict: Detection results containing:
                - annotated_image: Image with annotations
                - gaze_direction: Detected gaze direction
                - is_suspicious: Boolean indicating suspicious eye behavior
                - confidence: Detection confidence
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        annotated_image = image.copy()
        gaze_direction = "No Eyes Detected"
        is_suspicious = True
        confidence = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                img_h, img_w, _ = image.shape
                
                # Extract left eye landmarks
                left_eye_coords = []
                for idx in self.LEFT_EYE[:6]:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                    left_eye_coords.append([x, y])
                
                # Extract right eye landmarks
                right_eye_coords = []
                for idx in self.RIGHT_EYE[:6]:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                    right_eye_coords.append([x, y])
                
                left_eye_coords = np.array(left_eye_coords)
                right_eye_coords = np.array(right_eye_coords)
                
                # Calculate Eye Aspect Ratio
                left_ear = self.calculate_eye_aspect_ratio(left_eye_coords)
                right_ear = self.calculate_eye_aspect_ratio(right_eye_coords)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Extract iris landmarks
                left_iris_coords = []
                for idx in self.LEFT_IRIS:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                    left_iris_coords.append([x, y])
                    cv2.circle(annotated_image, (x, y), 1, (0, 255, 255), -1)
                
                right_iris_coords = []
                for idx in self.RIGHT_IRIS:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                    right_iris_coords.append([x, y])
                    cv2.circle(annotated_image, (x, y), 1, (0, 255, 255), -1)
                
                # Get full eye landmarks
                left_eye_full = np.array([
                    [int(face_landmarks.landmark[idx].x * img_w),
                     int(face_landmarks.landmark[idx].y * img_h)]
                    for idx in self.LEFT_EYE
                ])
                
                right_eye_full = np.array([
                    [int(face_landmarks.landmark[idx].x * img_w),
                     int(face_landmarks.landmark[idx].y * img_h)]
                    for idx in self.RIGHT_EYE
                ])
                
                # Calculate gaze direction
                if len(left_iris_coords) > 0 and len(right_iris_coords) > 0:
                    left_h_ratio, left_v_ratio = self.get_iris_position(
                        np.array(left_iris_coords), left_eye_full
                    )
                    right_h_ratio, right_v_ratio = self.get_iris_position(
                        np.array(right_iris_coords), right_eye_full
                    )
                    
                    avg_h_ratio = (left_h_ratio + right_h_ratio) / 2.0
                    avg_v_ratio = (left_v_ratio + right_v_ratio) / 2.0
                    
                    # Determine gaze direction
                    if avg_h_ratio < (0.5 - self.gaze_threshold):
                        gaze_direction = "LEFT"
                        is_suspicious = True
                    elif avg_h_ratio > (0.5 + self.gaze_threshold):
                        gaze_direction = "RIGHT"
                        is_suspicious = True
                    elif avg_v_ratio < (0.5 - self.gaze_threshold):
                        gaze_direction = "UP"
                        is_suspicious = True
                    elif avg_v_ratio > (0.5 + self.gaze_threshold):
                        gaze_direction = "DOWN"
                        is_suspicious = True
                    else:
                        gaze_direction = "CENTER"
                        is_suspicious = False
                    
                    # Check for eye closure
                    if avg_ear < self.eye_aspect_ratio_threshold:
                        gaze_direction = "EYES CLOSED"
                        is_suspicious = True
                
                # Draw eye contours
                for coord in left_eye_full:
                    cv2.circle(annotated_image, tuple(coord), 1, (0, 255, 0), -1)
                for coord in right_eye_full:
                    cv2.circle(annotated_image, tuple(coord), 1, (0, 255, 0), -1)
                
                # Display information
                color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                status = "SUSPICIOUS" if is_suspicious else "NORMAL"
                
                cv2.putText(annotated_image, f"Gaze: {gaze_direction}", (20, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(annotated_image, f"Eye Status: {status}", (20, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                confidence = 0.95
        
        return {
            'annotated_image': annotated_image,
            'gaze_direction': gaze_direction,
            'is_suspicious': is_suspicious,
            'confidence': confidence
        }
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()
