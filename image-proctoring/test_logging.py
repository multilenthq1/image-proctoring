#!/usr/bin/env python3
"""
Test script to verify server logging functionality.
Sends a few test frames to the server and checks if logs are created.
"""

import requests
import base64
import cv2
import time
import os
import json

def test_server_logging():
    """Test server logging functionality."""
    server_url = "http://localhost:5550"
    
    print("="*60)
    print("SERVER LOGGING TEST")
    print("="*60)
    
    # 1. Check server health
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ Server is running")
        else:
            print("   ✗ Server returned unexpected status")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Cannot connect to server: {e}")
        print("   Please start the server: python server.py")
        return False
    
    # 2. Send test frames
    print("\n2. Sending test frames...")
    test_client_id = "test_client_001"
    
    # Create a simple test image
    test_image = cv2.imread("sample_images/webcam_capture.jpg")
    if test_image is None:
        # Create a blank test image if sample doesn't exist
        test_image = cv2.imread("sample_images/README.md")
        if test_image is None:
            print("   ⚠ No sample image found, creating blank test image")
            test_image = (255 * cv2.randn((480, 640, 3))).astype('uint8')
    
    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send 5 test frames
    for i in range(1, 6):
        payload = {
            'image': image_base64,
            'return_annotated': False,
            'client_id': test_client_id
        }
        
        response = requests.post(f"{server_url}/analyze", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            status = "SUSPICIOUS" if result['is_suspicious'] else "NORMAL"
            print(f"   Frame {i}: {status} - {len(result.get('violations', []))} violations")
        else:
            print(f"   ✗ Frame {i} failed: {response.status_code}")
        
        time.sleep(0.5)  # Small delay between frames
    
    # 3. Check statistics
    print("\n3. Checking server statistics...")
    response = requests.get(f"{server_url}/stats", timeout=5)
    if response.status_code == 200:
        stats = response.json()
        print(f"   Total frames analyzed: {stats['total_frames_analyzed']}")
        print(f"   Suspicious frames: {stats['suspicious_frames']}")
        print(f"   Normal frames: {stats['normal_frames']}")
        print(f"   Suspicious rate: {stats['suspicious_rate']:.1f}%")
        print(f"   Uptime: {stats['uptime_seconds']:.1f} seconds")
    else:
        print("   ✗ Failed to get statistics")
    
    # 4. Check log files
    print("\n4. Checking log files...")
    
    # Check server.log
    if os.path.exists("server_logs/server.log"):
        size = os.path.getsize("server_logs/server.log")
        print(f"   ✓ server.log exists ({size} bytes)")
        
        # Show last few lines
        with open("server_logs/server.log", 'r') as f:
            lines = f.readlines()
            print("\n   Last 5 log entries:")
            for line in lines[-5:]:
                print(f"     {line.strip()}")
    else:
        print("   ✗ server.log not found")
    
    # Check analysis logs
    print("\n   Analysis log files:")
    if os.path.exists("server_logs"):
        json_files = [f for f in os.listdir("server_logs") if f.startswith("analysis_log_")]
        if json_files:
            for json_file in json_files:
                size = os.path.getsize(f"server_logs/{json_file}")
                print(f"     ✓ {json_file} ({size} bytes)")
                
                # Parse latest log
                with open(f"server_logs/{json_file}", 'r') as f:
                    data = json.load(f)
                    print(f"       - {data['total_frames']} frames logged")
                    print(f"       - {data['suspicious_frames']} suspicious")
        else:
            print("     ⚠ No analysis log files yet (created every 10 frames or on violations)")
    else:
        print("   ✗ server_logs directory not found")
    
    print("\n" + "="*60)
    print("LOGGING TEST COMPLETE")
    print("="*60)
    print("\nTo view live logs, run:")
    print("  tail -f server_logs/server.log")
    print("\nTo view violations only:")
    print("  tail -f server_logs/server.log | grep WARNING")
    print("="*60)
    
    return True


if __name__ == "__main__":
    test_server_logging()
