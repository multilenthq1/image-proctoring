"""
Image Proctoring Application
Analyzes exam candidate images for potential violations.
"""

import cv2
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Import proctoring modules
from src.head_pose_detector import HeadPoseDetector
from src.eye_tracker import EyeTracker
from src.object_detector import ObjectDetector


class ImageProctoringApp:
    """
    Image proctoring application that analyzes static images.
    """
    
    def __init__(self):
        """Initialize all proctoring modules."""
        print("Initializing Image Proctoring Application...")
        
        # Initialize detectors
        self.head_pose_detector = HeadPoseDetector()
        self.eye_tracker = EyeTracker()
        self.object_detector = ObjectDetector()
        
        print("Initialization complete!")
    
    def analyze_image(self, image_path, save_annotated=True, output_dir="results"):
        """
        Analyze a single image for proctoring violations.
        
        Args:
            image_path: Path to the image file
            save_annotated: Whether to save annotated images
            output_dir: Directory to save results
            
        Returns:
            dict: Analysis results
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            return None
        
        print(f"\nAnalyzing image: {image_path}")
        print("="*60)
        
        # Prepare results
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'violations': [],
            'analysis': {}
        }
        
        # Head Pose Detection
        print("Running head pose detection...")
        head_pose_result = self.head_pose_detector.detect_head_pose(image)
        results['analysis']['head_pose'] = {
            'direction': head_pose_result['direction'],
            'is_suspicious': head_pose_result['is_suspicious'],
            'confidence': head_pose_result['confidence']
        }
        
        if head_pose_result['is_suspicious']:
            violation = f"Head pose violation: Looking {head_pose_result['direction']}"
            results['violations'].append(violation)
            print(f"  ⚠️  {violation}")
        else:
            print(f"  ✓  Head pose: {head_pose_result['direction']}")
        
        # Eye Tracking
        print("Running eye tracking...")
        eye_tracking_result = self.eye_tracker.track_eyes(image)
        results['analysis']['eye_tracking'] = {
            'gaze_direction': eye_tracking_result['gaze_direction'],
            'is_suspicious': eye_tracking_result['is_suspicious'],
            'confidence': eye_tracking_result['confidence']
        }
        
        if eye_tracking_result['is_suspicious']:
            violation = f"Eye tracking violation: Gaze {eye_tracking_result['gaze_direction']}"
            results['violations'].append(violation)
            print(f"  ⚠️  {violation}")
        else:
            print(f"  ✓  Gaze direction: {eye_tracking_result['gaze_direction']}")
        
        # Object Detection
        print("Running object detection...")
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
                print(f"  ⚠️  {violation}")
            
            if detection_info['phone_detected']:
                violation = "Object violation: Phone detected"
                results['violations'].append(violation)
                print(f"  ⚠️  {violation}")
            
            if detection_info['detected_objects']:
                violation = f"Object violation: {', '.join(detection_info['detected_objects'])}"
                results['violations'].append(violation)
                print(f"  ⚠️  {violation}")
            
            if detection_info['num_hands'] > 2:
                violation = f"Hand count violation: {detection_info['num_hands']} hands detected"
                results['violations'].append(violation)
                print(f"  ⚠️  {violation}")
        else:
            print(f"  ✓  Object detection: Normal")
            print(f"     - Faces: {detection_info['num_faces']}")
            print(f"     - Hands: {detection_info['num_hands']}")
        
        # Overall verdict
        results['is_suspicious'] = len(results['violations']) > 0
        
        print("\n" + "="*60)
        if results['is_suspicious']:
            print(f"⚠️  ANALYSIS RESULT: SUSPICIOUS - {len(results['violations'])} violation(s) detected")
        else:
            print("✓  ANALYSIS RESULT: NORMAL - No violations detected")
        print("="*60)
        
        # Save annotated images if requested
        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename
            base_name = Path(image_path).stem
            
            # Create combined annotated image
            combined_image = image.copy()
            
            # Apply all annotations
            combined_image = head_pose_result['annotated_image']
            combined_image = eye_tracking_result['annotated_image']
            combined_image = object_detection_result['annotated_image']
            
            # Save combined result
            output_path = os.path.join(output_dir, f"{base_name}_analyzed.jpg")
            cv2.imwrite(output_path, combined_image)
            results['output_image'] = output_path
            print(f"\nAnnotated image saved: {output_path}")
            
            # Also save individual analysis images
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_head_pose.jpg"), 
                       head_pose_result['annotated_image'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_eye_tracking.jpg"), 
                       eye_tracking_result['annotated_image'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_object_detection.jpg"), 
                       object_detection_result['annotated_image'])
            
            # Save JSON report
            json_path = os.path.join(output_dir, f"{base_name}_report.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"JSON report saved: {json_path}")
        
        return results
    
    def analyze_batch(self, image_folder, output_dir="results"):
        """
        Analyze multiple images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            output_dir: Directory to save results
            
        Returns:
            list: List of analysis results
        """
        # Check if folder exists
        if not os.path.exists(image_folder):
            print(f"Error: Folder not found: {image_folder}")
            return []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f"*{ext}"))
            image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in: {image_folder}")
            return []
        
        print(f"\nFound {len(image_files)} image(s) to analyze")
        print("="*60)
        
        # Analyze each image
        all_results = []
        suspicious_count = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            result = self.analyze_image(str(image_path), save_annotated=True, output_dir=output_dir)
            
            if result:
                all_results.append(result)
                if result['is_suspicious']:
                    suspicious_count += 1
        
        # Generate summary report
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total images analyzed: {len(all_results)}")
        print(f"Suspicious images: {suspicious_count}")
        print(f"Normal images: {len(all_results) - suspicious_count}")
        print("="*60)
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_images': len(all_results),
            'suspicious_count': suspicious_count,
            'normal_count': len(all_results) - suspicious_count,
            'results': all_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nBatch summary saved: {summary_path}")
        
        return all_results
    
    def cleanup(self):
        """Release all resources."""
        print("\nCleaning up resources...")
        self.head_pose_detector.release()
        self.eye_tracker.release()
        self.object_detector.release()
        print("Cleanup complete!")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("IMAGE PROCTORING APPLICATION")
    print("="*60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single image:  python analyze_image.py <image_path>")
        print("  Batch mode:    python analyze_image.py --batch <folder_path>")
        print("\nExample:")
        print("  python analyze_image.py sample_images/candidate1.jpg")
        print("  python analyze_image.py --batch sample_images/")
        sys.exit(1)
    
    # Initialize app
    app = ImageProctoringApp()
    
    try:
        # Check for batch mode
        if sys.argv[1] == "--batch":
            if len(sys.argv) < 3:
                print("Error: Please provide folder path for batch mode")
                sys.exit(1)
            
            folder_path = sys.argv[2]
            app.analyze_batch(folder_path)
        else:
            # Single image mode
            image_path = sys.argv[1]
            app.analyze_image(image_path)
    
    finally:
        # Cleanup
        app.cleanup()


if __name__ == "__main__":
    main()
