"""
Generate sample test images for the proctoring system
"""

import cv2
import numpy as np
import urllib.request
import os


def download_sample_image():
    """Download a sample person image from a public source."""
    # Using a sample image from a public dataset
    sample_urls = [
        # Sample faces from public datasets
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
    ]
    
    output_dir = "sample_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading sample image...")
    try:
        # Download first available image
        url = sample_urls[0]
        output_path = os.path.join(output_dir, "test_sample.jpg")
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Sample image downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def generate_synthetic_test_image():
    """Generate a synthetic test image with shapes and patterns."""
    # Create a blank image (simulating a webcam capture)
    width, height = 1280, 720
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add a simulated "person" area (oval for face)
    center_x, center_y = width // 2, height // 2 - 50
    face_width, face_height = 200, 250
    
    # Draw face oval
    cv2.ellipse(image, (center_x, center_y), (face_width // 2, face_height // 2), 
                0, 0, 360, (220, 180, 160), -1)
    
    # Draw eyes
    left_eye_x = center_x - 50
    right_eye_x = center_x + 50
    eye_y = center_y - 30
    cv2.ellipse(image, (left_eye_x, eye_y), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (right_eye_x, eye_y), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(image, (left_eye_x, eye_y), 8, (50, 50, 50), -1)
    cv2.circle(image, (right_eye_x, eye_y), 8, (50, 50, 50), -1)
    
    # Draw nose
    nose_points = np.array([
        [center_x, center_y],
        [center_x - 10, center_y + 40],
        [center_x + 10, center_y + 40]
    ], np.int32)
    cv2.polylines(image, [nose_points], True, (180, 140, 120), 2)
    
    # Draw mouth
    cv2.ellipse(image, (center_x, center_y + 70), (40, 20), 0, 0, 180, (180, 100, 100), 2)
    
    # Add hair
    cv2.ellipse(image, (center_x, center_y - 80), (face_width // 2 + 20, 80), 
                0, 180, 360, (80, 60, 40), -1)
    
    # Add text label
    cv2.putText(image, "SYNTHETIC TEST IMAGE", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "Generated for testing proctoring system", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Add a simulated desk/background
    cv2.rectangle(image, (0, height - 200), (width, height), (160, 140, 120), -1)
    
    # Add a book-like object on the desk (suspicious object)
    book_x, book_y = 900, height - 150
    cv2.rectangle(image, (book_x, book_y), (book_x + 120, book_y + 80), (200, 50, 50), -1)
    cv2.rectangle(image, (book_x, book_y), (book_x + 120, book_y + 80), (0, 0, 0), 2)
    cv2.putText(image, "BOOK", (book_x + 25, book_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save the image
    output_dir = "sample_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_test.jpg")
    cv2.imwrite(output_path, image)
    print(f"✓ Synthetic test image generated: {output_path}")
    
    return output_path


def create_webcam_snapshot():
    """Try to capture an image from the webcam."""
    print("Attempting to capture from webcam...")
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return None
        
        # Wait a moment for camera to initialize
        import time
        time.sleep(1)
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Could not capture frame")
            return None
        
        # Save the image
        output_dir = "sample_images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "webcam_capture.jpg")
        cv2.imwrite(output_path, frame)
        print(f"✓ Webcam image captured: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"✗ Webcam capture failed: {e}")
        return None


def main():
    print("="*60)
    print("TEST IMAGE GENERATOR")
    print("="*60)
    print("\nGenerating test images for proctoring system...\n")
    
    generated_images = []
    
    # Option 1: Try to capture from webcam
    print("1. Trying webcam capture...")
    webcam_img = create_webcam_snapshot()
    if webcam_img:
        generated_images.append(webcam_img)
    
    # Option 2: Download sample image
    print("\n2. Downloading sample image...")
    downloaded_img = download_sample_image()
    if downloaded_img:
        generated_images.append(downloaded_img)
    
    # Option 3: Generate synthetic image
    print("\n3. Generating synthetic test image...")
    synthetic_img = generate_synthetic_test_image()
    if synthetic_img:
        generated_images.append(synthetic_img)
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(generated_images)} test image(s):\n")
    for img in generated_images:
        print(f"  ✓ {img}")
    
    print("\nYou can now test with:")
    print(f"  python analyze_image.py {generated_images[0] if generated_images else 'sample_images/test_image.jpg'}")
    print("\nOr batch process:")
    print("  python analyze_image.py --batch sample_images/")
    print("="*60)


if __name__ == "__main__":
    main()
