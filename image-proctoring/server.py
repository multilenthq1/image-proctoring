"""
Image Proctoring Server
Receives images via HTTP and analyzes them for violations.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler

# Import proctoring modules
from src.head_pose_detector import HeadPoseDetector
from src.eye_tracker import EyeTracker
from src.object_detector import ObjectDetector


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global proctoring instance
proctoring_system = None

# Setup logging
def setup_logging():
    """Configure logging for the server."""
    # Create logs directory
    os.makedirs('server_logs', exist_ok=True)
    
    # Setup file handler with rotation
    log_file = os.path.join('server_logs', 'server.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('proctoring_server')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


class ProctoringServer:
    """
    Server-side proctoring system that analyzes frames.
    """
    
    def __init__(self):
        """Initialize all proctoring modules."""
        print("Initializing Proctoring Server...")
        logger.info("Initializing Proctoring Server...")
        
        # Initialize detectors
        self.head_pose_detector = HeadPoseDetector()
        self.eye_tracker = EyeTracker()
        self.object_detector = ObjectDetector()
        
        # Statistics
        self.total_frames_analyzed = 0
        self.suspicious_frames = 0
        self.session_start = datetime.now()
        self.analysis_log = []
        
        print("Server initialization complete!")
        logger.info("Server initialization complete!")
    
    def analyze_frame(self, image, return_annotated=False, client_id=None):
        """
        Analyze a single frame for proctoring violations.
        
        Args:
            image: OpenCV image (numpy array)
            return_annotated: Whether to return annotated image
            client_id: Optional client identifier
            
        Returns:
            dict: Analysis results
        """
        # Update statistics
        self.total_frames_analyzed += 1
        
        # Prepare results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = {
            'timestamp': timestamp,
            'frame_id': self.total_frames_analyzed,
            'violations': [],
            'analysis': {}
        }
        
        if client_id:
            results['client_id'] = client_id
        
        logger.info(f"Analyzing frame {self.total_frames_analyzed}" + 
                   (f" from client {client_id}" if client_id else ""))
        
        # Head Pose Detection
        head_pose_result = self.head_pose_detector.detect_head_pose(image)
        results['analysis']['head_pose'] = {
            'direction': head_pose_result['direction'],
            'is_suspicious': head_pose_result['is_suspicious'],
            'confidence': head_pose_result['confidence']
        }
        
        if head_pose_result['is_suspicious']:
            violation = f"Head pose violation: Looking {head_pose_result['direction']}"
            results['violations'].append(violation)
        
        # Eye Tracking
        eye_tracking_result = self.eye_tracker.track_eyes(image)
        results['analysis']['eye_tracking'] = {
            'gaze_direction': eye_tracking_result['gaze_direction'],
            'is_suspicious': eye_tracking_result['is_suspicious'],
            'confidence': eye_tracking_result['confidence']
        }
        
        if eye_tracking_result['is_suspicious']:
            violation = f"Eye tracking violation: Gaze {eye_tracking_result['gaze_direction']}"
            results['violations'].append(violation)
        
        # Object Detection
        object_detection_result = self.object_detector.detect_objects(image)
        detection_info = object_detection_result['detection_info']
        
        results['analysis']['object_detection'] = {
            'num_faces': detection_info['num_faces'],
            'num_hands': detection_info['num_hands'],
            'phone_detected': detection_info['phone_detected'],
            'detected_objects': detection_info['detected_objects'],
            'is_suspicious': object_detection_result['is_suspicious']
        }
        
        if object_detection_result['is_suspicious']:
            if detection_info['num_faces'] != 1:
                violation = f"Face count violation: {detection_info['num_faces']} face(s) detected"
                results['violations'].append(violation)
            
            if detection_info['phone_detected']:
                violation = "Object violation: Phone detected"
                results['violations'].append(violation)
            
            if detection_info['detected_objects']:
                violation = f"Object violation: {', '.join(detection_info['detected_objects'])}"
                results['violations'].append(violation)
            
            if detection_info['num_hands'] > 2:
                violation = f"Hand count violation: {detection_info['num_hands']} hands detected"
                results['violations'].append(violation)
        
        # Overall verdict
        results['is_suspicious'] = len(results['violations']) > 0
        
        # Update statistics
        if results['is_suspicious']:
            self.suspicious_frames += 1
            logger.warning(f"Frame {self.total_frames_analyzed}: SUSPICIOUS - {len(results['violations'])} violations")
            for violation in results['violations']:
                logger.warning(f"  - {violation}")
        else:
            logger.info(f"Frame {self.total_frames_analyzed}: Normal")
        
        # Log the analysis
        self.analysis_log.append({
            'frame_id': self.total_frames_analyzed,
            'timestamp': timestamp,
            'is_suspicious': results['is_suspicious'],
            'violations': results['violations'],
            'client_id': client_id
        })
        
        # Save detailed log every 10 frames or if suspicious
        if self.total_frames_analyzed % 10 == 0 or results['is_suspicious']:
            self._save_analysis_log()
        
        # Add annotated image if requested
        if return_annotated:
            # Use the object detection annotated image as it has all overlays
            annotated = object_detection_result['annotated_image']
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')
            results['annotated_image'] = annotated_base64
        
        return results
    
    def _save_analysis_log(self):
        """Save analysis log to file."""
        try:
            log_filename = f"analysis_log_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join('server_logs', log_filename)
            
            log_data = {
                'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
                'total_frames': self.total_frames_analyzed,
                'suspicious_frames': self.suspicious_frames,
                'normal_frames': self.total_frames_analyzed - self.suspicious_frames,
                'analysis_log': self.analysis_log
            }
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save analysis log: {str(e)}")
    
    def get_statistics(self):
        """Get server statistics."""
        uptime = (datetime.now() - self.session_start).total_seconds()
        return {
            'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'uptime_seconds': uptime,
            'total_frames_analyzed': self.total_frames_analyzed,
            'suspicious_frames': self.suspicious_frames,
            'normal_frames': self.total_frames_analyzed - self.suspicious_frames,
            'suspicious_rate': (self.suspicious_frames / self.total_frames_analyzed * 100) 
                              if self.total_frames_analyzed > 0 else 0
        }
    
    def cleanup(self):
        """Release all resources."""
        print("Cleaning up server resources...")
        logger.info("Cleaning up server resources...")
        
        # Save final analysis log
        self._save_analysis_log()
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"Session statistics:")
        logger.info(f"  Total frames analyzed: {stats['total_frames_analyzed']}")
        logger.info(f"  Suspicious frames: {stats['suspicious_frames']}")
        logger.info(f"  Normal frames: {stats['normal_frames']}")
        logger.info(f"  Uptime: {stats['uptime_seconds']:.1f} seconds")
        
        self.head_pose_detector.release()
        self.eye_tracker.release()
        self.object_detector.release()
        print("Server cleanup complete!")
        logger.info("Server cleanup complete!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics."""
    if proctoring_system:
        stats = proctoring_system.get_statistics()
        logger.info("Statistics requested")
        return jsonify(stats), 200
    else:
        return jsonify({'error': 'Server not initialized'}), 500


@app.route('/analyze', methods=['POST'])
def analyze_frame():
    """
    Analyze a frame sent from the client.
    
    Expected JSON format:
    {
        "image": "base64_encoded_image",
        "return_annotated": false  # optional
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        client_id = data.get('client_id') if data else None
        
        if not data or 'image' not in data:
            logger.warning("Analyze request with no image data")
            return jsonify({
                'error': 'No image data provided'
            }), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            return jsonify({
                'error': 'Failed to decode image'
            }), 400
        
        # Get return_annotated flag
        return_annotated = data.get('return_annotated', False)
        
        # Analyze the frame
        results = proctoring_system.analyze_frame(image, return_annotated, client_id)
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}", exc_info=True)
        print(f"Error analyzing frame: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple frames sent from the client.
    
    Expected JSON format:
    {
        "images": ["base64_1", "base64_2", ...],
        "return_annotated": false  # optional
    }
    """
    try:
        data = request.get_json()
        client_id = data.get('client_id') if data else None
        
        if not data or 'images' not in data:
            logger.warning("Batch analyze request with no images data")
            return jsonify({
                'error': 'No images data provided'
            }), 400
        
        images_data = data['images']
        return_annotated = data.get('return_annotated', False)
        
        logger.info(f"Batch analysis started: {len(images_data)} frames")
        
        results = []
        for i, img_data in enumerate(images_data, 1):
            # Decode base64 image
            image_data = base64.b64decode(img_data)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                result = proctoring_system.analyze_frame(image, return_annotated, client_id)
                results.append(result)
        
        logger.info(f"Batch analysis completed: {len(results)} frames processed")
        
        return jsonify({
            'batch_results': results,
            'total_frames': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing batch: {str(e)}", exc_info=True)
        print(f"Error analyzing batch: {str(e)}")
        return jsonify({
            'error': f'Batch analysis failed: {str(e)}'
        }), 500


def initialize_server():
    """Initialize the proctoring server."""
    global proctoring_system
    proctoring_system = ProctoringServer()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("IMAGE PROCTORING SERVER")
    print("="*60)
    
    # Initialize server
    initialize_server()
    
    # Start Flask server
    print("\nStarting server on http://0.0.0.0:5550")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5550, debug=False, threaded=True)
    finally:
        # Cleanup on exit
        if proctoring_system:
            proctoring_system.cleanup()
