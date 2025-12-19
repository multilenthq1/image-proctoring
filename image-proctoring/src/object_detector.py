"""
Object Detection Module - Image Version
Detects unauthorized objects and multiple faces in static images.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request


class ObjectDetector:
    """
    Detects objects in images to identify potential cheating aids.
    """
    
    def __init__(self):
        """Initialize MediaPipe detection modules."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize Object Detector
        self.object_detector = None
        self._init_object_detector()
        
        # Suspicious object categories
        self.suspicious_categories = ['cell phone', 'book', 'laptop', 'remote']
        
        print("Object Detector initialized!")
    
    def _init_object_detector(self):
        """Initialize MediaPipe Tasks Object Detector."""
        model_path = "efficientdet_lite0.tflite"
        
        # Download model if not present
        if not os.path.exists(model_path):
            print("Downloading EfficientDet model...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
                    model_path
                )
                print("Model downloaded successfully!")
            except Exception as e:
                print(f"Could not download model: {e}")
                return
        
        # Create object detector
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=0.5,
                max_results=5
            )
            self.object_detector = vision.ObjectDetector.create_from_options(options)
            print("Object Detector loaded!")
        except Exception as e:
            print(f"Could not initialize object detector: {e}")
            self.object_detector = None
    
    def detect_objects_efficientdet(self, image):
        """Detect objects using MediaPipe EfficientDet."""
        detected_objects = []
        phone_detected = False
        annotated_image = image.copy()
        
        if self.object_detector is None:
            return annotated_image, detected_objects, phone_detected
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect objects
        detection_result = self.object_detector.detect(mp_image)
        
        # Process detections
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            category_name = category.category_name
            score = category.score
            
            # Check if suspicious
            if category_name.lower() in [s.lower() for s in self.suspicious_categories]:
                detected_objects.append(category_name)
                
                if category_name.lower() == 'cell phone':
                    phone_detected = True
                
                # Draw bounding box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(annotated_image, start_point, end_point, (0, 0, 255), 3)
                
                # Draw label
                label = f"{category_name.upper()} ({score:.2f})"
                cv2.putText(annotated_image, label, (bbox.origin_x, bbox.origin_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if category_name.lower() == 'cell phone':
                    cv2.putText(annotated_image, "PHONE DETECTED!", (bbox.origin_x, bbox.origin_y - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated_image, detected_objects, phone_detected
    
    def detect_hands(self, image):
        """Detect hands in the image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        num_hands = 0
        hand_positions = []
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Get hand position
                img_h, img_w, _ = image.shape
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                x, y = int(wrist.x * img_w), int(wrist.y * img_h)
                hand_positions.append((x, y))
                
                # Label the hand
                hand_label = handedness.classification[0].label
                cv2.putText(annotated_image, hand_label, (x - 30, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image, num_hands, hand_positions
    
    def detect_faces(self, image):
        """Detect faces using MediaPipe Face Detection."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        num_faces = 0
        is_suspicious = False
        annotated_image = image.copy()
        
        img_h, img_w, _ = image.shape
        
        if results.detections:
            num_faces = len(results.detections)
            is_suspicious = num_faces != 1
            
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * img_w)
                y = int(bboxC.ymin * img_h)
                w = int(bboxC.width * img_w)
                h = int(bboxC.height * img_h)
                
                color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                
                confidence = detection.score[0]
                cv2.putText(annotated_image, f"Face: {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            is_suspicious = True
        
        return annotated_image, num_faces, is_suspicious
    
    def detect_objects(self, image):
        """
        Comprehensive object detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            dict: Detection results containing:
                - annotated_image: Image with all annotations
                - detection_info: Dictionary with detection details
                - is_suspicious: Boolean indicating suspicious activity
        """
        # Start with original image
        annotated_image = image.copy()
        
        # Detect objects (phones, books, laptops)
        annotated_image, detected_objects, phone_detected = self.detect_objects_efficientdet(annotated_image)
        
        # Detect faces
        annotated_image, num_faces, faces_suspicious = self.detect_faces(annotated_image)
        
        # Detect hands
        annotated_image, num_hands, hand_positions = self.detect_hands(annotated_image)
        
        # Overall suspicious status
        is_suspicious = faces_suspicious or num_hands > 2 or phone_detected or len(detected_objects) > 0
        
        # Prepare detection info
        detection_info = {
            'num_hands': num_hands,
            'num_faces': num_faces,
            'num_objects': len(detected_objects),
            'hand_positions': hand_positions,
            'phone_detected': phone_detected,
            'detected_objects': detected_objects
        }
        
        # Display detection information
        y_offset = 130
        
        # Faces
        face_color = (0, 0, 255) if faces_suspicious else (0, 255, 0)
        face_status = "SUSPICIOUS" if faces_suspicious else "OK"
        cv2.putText(annotated_image, f"Faces: {num_faces} ({face_status})", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Hands
        y_offset += 30
        hand_color = (0, 0, 255) if num_hands > 2 else (0, 255, 0)
        cv2.putText(annotated_image, f"Hands: {num_hands}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Phone detection
        y_offset += 30
        phone_color = (0, 0, 255) if phone_detected else (0, 255, 0)
        phone_status = "DETECTED!" if phone_detected else "None"
        cv2.putText(annotated_image, f"Phone: {phone_status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2)
        
        # Other suspicious objects
        if detected_objects:
            y_offset += 30
            objects_str = ", ".join(detected_objects)
            cv2.putText(annotated_image, f"Objects: {objects_str}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Overall status
        y_offset += 30
        status_color = (0, 0, 255) if is_suspicious else (0, 255, 0)
        status = "SUSPICIOUS" if is_suspicious else "NORMAL"
        cv2.putText(annotated_image, f"Status: {status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return {
            'annotated_image': annotated_image,
            'detection_info': detection_info,
            'is_suspicious': is_suspicious
        }
    
    def release(self):
        """Release resources."""
        self.hands.close()
        self.face_detection.close()
        if self.object_detector is not None:
            self.object_detector.close()
