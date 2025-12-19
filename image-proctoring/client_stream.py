"""
Video Streaming Client
Captures video frames and sends them to the proctoring server for analysis.
"""

import cv2
import base64
import requests
import json
import time
import argparse
from datetime import datetime
import os


class VideoStreamClient:
    """
    Client that streams video frames to the proctoring server.
    """
    
    def __init__(self, server_url="http://localhost:5550", video_source=0, fps=1, client_id=None):
        """
        Initialize the video streaming client.
        
        Args:
            server_url: URL of the proctoring server
            video_source: Video source (0 for webcam, or path to video file)
            fps: Frames per second to send to server (default: 1 frame/sec)
            client_id: Optional client identifier for tracking
        """
        self.server_url = server_url
        self.video_source = video_source
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.client_id = client_id or f"client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Statistics
        self.frames_sent = 0
        self.frames_suspicious = 0
        self.total_violations = []
        
        print(f"Initializing Video Stream Client...")
        print(f"  Client ID: {self.client_id}")
        print(f"  Server: {self.server_url}")
        print(f"  Video Source: {self.video_source}")
        print(f"  FPS: {self.fps}")
    
    def check_server_health(self):
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Server is healthy and ready")
                return True
            else:
                print(f"✗ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to server: {str(e)}")
            print(f"  Make sure the server is running at {self.server_url}")
            return False
    
    def encode_frame(self, frame):
        """Encode frame to base64."""
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    
    def send_frame(self, frame, return_annotated=False):
        """
        Send a frame to the server for analysis.
        
        Args:
            frame: OpenCV frame (numpy array)
            return_annotated: Whether to request annotated image
            
        Returns:
            dict: Analysis results or None if failed
        """
        try:
            # Encode frame
            frame_base64 = self.encode_frame(frame)
            
            # Prepare request
            payload = {
                'image': frame_base64,
                'return_annotated': return_annotated,
                'client_id': self.client_id
            }
            
            # Send to server
            response = requests.post(
                f"{self.server_url}/analyze",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Server error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
    
    def display_results(self, results, frame):
        """
        Display analysis results on the frame.
        
        Args:
            results: Analysis results from server
            frame: OpenCV frame to annotate
        """
        if not results:
            return frame
        
        # Create overlay text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Status
        status = "SUSPICIOUS" if results['is_suspicious'] else "NORMAL"
        color = (0, 0, 255) if results['is_suspicious'] else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 30
        
        # Violations
        if results['violations']:
            cv2.putText(frame, f"Violations: {len(results['violations'])}", 
                       (10, y_offset), font, font_scale, (0, 0, 255), thickness)
            y_offset += 25
            
            for violation in results['violations'][:3]:  # Show max 3 violations
                cv2.putText(frame, f"  - {violation[:50]}", (10, y_offset), 
                           font, 0.5, (0, 0, 255), 1)
                y_offset += 20
        
        # Analysis details
        analysis = results.get('analysis', {})
        
        # Head pose
        if 'head_pose' in analysis:
            head_pose = analysis['head_pose']
            cv2.putText(frame, f"Head: {head_pose['direction']}", 
                       (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Eye tracking
        if 'eye_tracking' in analysis:
            eye_tracking = analysis['eye_tracking']
            cv2.putText(frame, f"Gaze: {eye_tracking['gaze_direction']}", 
                       (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Object detection
        if 'object_detection' in analysis:
            obj_det = analysis['object_detection']
            cv2.putText(frame, f"Faces: {obj_det['num_faces']} | Hands: {obj_det['num_hands']}", 
                       (10, y_offset), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def stream_video(self, display=True, save_log=True):
        """
        Start streaming video to the server.
        
        Args:
            display: Whether to display the video feed
            save_log: Whether to save analysis log
        """
        # Check server health
        if not self.check_server_health():
            print("\nPlease start the server first by running: python server.py")
            return
        
        # Open video capture
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.video_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nVideo opened: {width}x{height}")
        
        print("\n" + "="*60)
        print("STREAMING VIDEO TO SERVER")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 's' to save current frame analysis")
        print("="*60 + "\n")
        
        # Initialize log
        session_log = []
        session_start = datetime.now()
        
        last_frame_time = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video or cannot read frame")
                    break
                
                current_time = time.time()
                
                # Send frame at specified FPS
                if current_time - last_frame_time >= self.frame_interval:
                    last_frame_time = current_time
                    
                    # Send frame to server
                    print(f"Sending frame {self.frames_sent + 1}...", end=' ')
                    results = self.send_frame(frame, return_annotated=False)
                    
                    if results:
                        self.frames_sent += 1
                        
                        # Update statistics
                        if results['is_suspicious']:
                            self.frames_suspicious += 1
                            print("⚠️  SUSPICIOUS")
                            
                            # Collect violations
                            for violation in results['violations']:
                                if violation not in self.total_violations:
                                    self.total_violations.append(violation)
                        else:
                            print("✓  Normal")
                        
                        # Add to log
                        if save_log:
                            log_entry = {
                                'frame_number': self.frames_sent,
                                'timestamp': results['timestamp'],
                                'is_suspicious': results['is_suspicious'],
                                'violations': results['violations'],
                                'analysis': results['analysis']
                            }
                            session_log.append(log_entry)
                        
                        # Display results on frame
                        if display:
                            frame = self.display_results(results, frame)
                
                # Display frame
                if display:
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {self.frames_sent}", 
                               (width - 150, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Proctoring Stream', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopping stream...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n\nStream interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "="*60)
            print("SESSION STATISTICS")
            print("="*60)
            session_duration = (datetime.now() - session_start).total_seconds()
            print(f"Duration: {session_duration:.1f} seconds")
            print(f"Frames sent: {self.frames_sent}")
            print(f"Suspicious frames: {self.frames_suspicious}")
            print(f"Normal frames: {self.frames_sent - self.frames_suspicious}")
            
            if self.frames_sent > 0:
                suspicious_rate = (self.frames_suspicious / self.frames_sent) * 100
                print(f"Suspicious rate: {suspicious_rate:.1f}%")
            
            if self.total_violations:
                print(f"\nUnique violations detected:")
                for violation in self.total_violations:
                    print(f"  - {violation}")
            
            print("="*60)
            
            # Save log file
            if save_log and session_log:
                log_filename = f"stream_session_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
                os.makedirs("logs", exist_ok=True)
                log_path = os.path.join("logs", log_filename)
                
                session_summary = {
                    'session_start': session_start.strftime("%Y-%m-%d %H:%M:%S"),
                    'session_end': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'duration_seconds': session_duration,
                    'total_frames': self.frames_sent,
                    'suspicious_frames': self.frames_suspicious,
                    'normal_frames': self.frames_sent - self.frames_suspicious,
                    'unique_violations': self.total_violations,
                    'frame_log': session_log
                }
                
                with open(log_path, 'w') as f:
                    json.dump(session_summary, f, indent=2)
                
                print(f"\nSession log saved: {log_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Video Streaming Client for Proctoring')
    parser.add_argument('--server', type=str, default='http://localhost:5550',
                       help='Server URL (default: http://localhost:5550)')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam, or path to video file (default: 0)')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames per second to send (default: 1.0)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying video')
    parser.add_argument('--no-log', action='store_true',
                       help='Run without saving log')
    parser.add_argument('--client-id', type=str, default=None,
                       help='Client identifier for tracking (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    # Create client
    client = VideoStreamClient(
        server_url=args.server,
        video_source=video_source,
        fps=args.fps,
        client_id=args.client_id
    )
    
    # Start streaming
    client.stream_video(
        display=not args.no_display,
        save_log=not args.no_log
    )


if __name__ == "__main__":
    main()
