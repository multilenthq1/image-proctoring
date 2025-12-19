# Image Proctoring System - Client-Server Setup

## Overview
The system has been restructured into a client-server architecture:
- **Server**: Receives and analyzes video frames for proctoring violations
- **Client**: Captures video from webcam/video file and streams frames to the server

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python server.py
```
The server will start on `http://localhost:5000`

### 3. Run the Client
In a separate terminal, run:
```bash
python client_stream.py
```

## Usage

### Server (server.py)
The server provides REST API endpoints for frame analysis:

**Endpoints:**
- `GET /health` - Check server health
- `POST /analyze` - Analyze a single frame
- `POST /analyze_batch` - Analyze multiple frames

The server automatically initializes all proctoring modules (head pose detection, eye tracking, object detection).

### Client (client_stream.py)

**Basic usage:**
```bash
# Stream from webcam at 1 FPS
python client_stream.py

# Custom server URL
python client_stream.py --server http://192.168.1.100:5000

# Stream from video file
python client_stream.py --source path/to/video.mp4

# Adjust frame rate (2 frames per second)
python client_stream.py --fps 2.0

# Run without display window
python client_stream.py --no-display

# Run without saving logs
python client_stream.py --no-log
```

**Interactive Controls:**
- Press `q` to quit
- Press `s` to save current frame

**Features:**
- Real-time video streaming to server
- Configurable frame rate (FPS)
- Live display with analysis overlay
- Session logging in JSON format
- Statistics summary at the end

## Output

### Client Display
The client shows:
- Live video feed with analysis results
- Status (NORMAL/SUSPICIOUS)
- Violation count and details
- Head pose direction
- Gaze direction
- Face and hand count

### Session Logs
Logs are saved in the `logs/` directory with format:
```
logs/stream_session_YYYYMMDD_HHMMSS.json
```

Contains:
- Session statistics
- Frame-by-frame analysis
- Violation summary
- Timestamps

## Architecture

```
┌─────────────┐                    ┌─────────────┐
│   Client    │                    │   Server    │
│             │                    │             │
│ • Webcam    │  ──── HTTP ────>   │ • Flask API │
│ • Video     │   (Base64 JPG)     │ • Analysis  │
│   File      │                    │   Modules   │
│             │  <─── JSON ────    │             │
│ • Display   │   (Results)        │             │
└─────────────┘                    └─────────────┘
```

## API Format

### Request (POST /analyze)
```json
{
  "image": "base64_encoded_jpeg_string",
  "return_annotated": false
}
```

### Response
```json
{
  "timestamp": "2025-12-19 10:30:45",
  "is_suspicious": true,
  "violations": [
    "Head pose violation: Looking Left",
    "Eye tracking violation: Gaze Left"
  ],
  "analysis": {
    "head_pose": {
      "direction": "Left",
      "is_suspicious": true,
      "confidence": 0.95
    },
    "eye_tracking": {
      "gaze_direction": "Left",
      "is_suspicious": true,
      "confidence": 0.88
    },
    "object_detection": {
      "num_faces": 1,
      "num_hands": 2,
      "phone_detected": false,
      "detected_objects": [],
      "is_suspicious": false
    }
  }
}
```

## Performance Tips

1. **Frame Rate**: Lower FPS (0.5-2.0) is usually sufficient for proctoring
2. **Network**: Use localhost for best performance, LAN for remote monitoring
3. **Resources**: Server requires more CPU/memory for analysis modules
4. **Display**: Use `--no-display` on client for better performance

## Troubleshooting

**Server won't start:**
- Check if port 5000 is already in use
- Ensure all dependencies are installed

**Client can't connect:**
- Verify server is running: `curl http://localhost:5000/health`
- Check firewall settings for network access

**Webcam not working:**
- Try different source numbers: `--source 1` or `--source 2`
- On macOS, grant camera permissions to Terminal

**Slow performance:**
- Reduce FPS: `--fps 0.5`
- Use `--no-display` mode
- Ensure good network connection

## Original File

The original single-file implementation is still available in `analyze_image.py` for analyzing static images from disk.
