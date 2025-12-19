# Image Proctoring Application

A Python-based exam proctoring system that analyzes **static images** (not video streams) to detect potential violations during exams.

## Features

- **Head Pose Detection**: Analyzes head orientation to determine if candidate is looking at the screen
- **Eye Tracking**: Detects gaze direction to identify if eyes are looking away
- **Object Detection**: Identifies unauthorized objects (phones, books, laptops) in the frame
- **Face Detection**: Verifies exactly one face is present
- **Hand Detection**: Counts hands visible in the image
- **Detailed Reports**: Generates JSON reports and annotated images for each analysis

## Differences from Video Stream Version

| Feature | Video Stream | Image Analysis |
|---------|--------------|----------------|
| Input | Live camera feed | Static image files |
| Processing | Real-time continuous | Single/batch analysis |
| Noise Detection | ✓ (audio monitoring) | ✗ (not applicable) |
| Violation Logging | Continuous with timestamps | Per-image report |
| Output | Live display + log file | Annotated images + JSON reports |

## Installation

1. Navigate to the image-proctoring directory:
```bash
cd image-proctoring
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the EfficientDet model (optional - will auto-download on first run):
```bash
# Model will be downloaded automatically when running the application
# Or manually download from:
# https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
```

## Usage

### Single Image Analysis

Analyze a single image:
```bash
python analyze_image.py path/to/image.jpg
```

Example:
```bash
python analyze_image.py sample_images/candidate1.jpg
```

### Batch Analysis

Analyze all images in a folder:
```bash
python analyze_image.py --batch path/to/image/folder/
```

Example:
```bash
python analyze_image.py --batch sample_images/
```

## Output

### Generated Files

For each analyzed image, the system creates:

1. **Annotated Images** (in `results/` folder):
   - `{filename}_analyzed.jpg` - Combined analysis with all detections
   - `{filename}_head_pose.jpg` - Head pose analysis only
   - `{filename}_eye_tracking.jpg` - Eye tracking analysis only
   - `{filename}_object_detection.jpg` - Object detection analysis only

2. **JSON Report** (`{filename}_report.json`):
```json
{
  "image_path": "sample.jpg",
  "timestamp": "2025-12-19 14:30:00",
  "is_suspicious": true,
  "violations": [
    "Head pose violation: Looking Left",
    "Object violation: Phone detected"
  ],
  "analysis": {
    "head_pose": {
      "direction": "Left",
      "is_suspicious": true,
      "confidence": 0.95
    },
    "eye_tracking": {
      "gaze_direction": "CENTER",
      "is_suspicious": false,
      "confidence": 0.95
    },
    "object_detection": {
      "num_faces": 1,
      "num_hands": 2,
      "phone_detected": true,
      "detected_objects": ["cell phone"],
      "is_suspicious": true
    }
  }
}
```

3. **Batch Summary** (for batch mode - `batch_summary.json`):
```json
{
  "timestamp": "2025-12-19 14:30:00",
  "total_images": 10,
  "suspicious_count": 3,
  "normal_count": 7,
  "results": [...]
}
```

## Project Structure

```
image-proctoring/
├── analyze_image.py              # Main application script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── src/
│   ├── __init__.py
│   ├── head_pose_detector.py    # Head pose detection module
│   ├── eye_tracker.py           # Eye tracking module
│   └── object_detector.py       # Object detection module
├── sample_images/               # Place test images here
└── results/                     # Output directory (auto-created)
    ├── image1_analyzed.jpg
    ├── image1_report.json
    └── batch_summary.json
```

## Detection Details

### Head Pose Detection
- **Normal**: Looking forward at the screen
- **Suspicious**: Looking left, right, up, down, or combinations

### Eye Tracking
- **Normal**: Gaze centered on screen
- **Suspicious**: Looking left, right, up, down, or eyes closed

### Object Detection
- **Detected Objects**: Cell phones, books, laptops, remotes
- **Face Count**: Should be exactly 1 face
- **Hand Count**: Flagged if more than 2 hands detected

## Example Output

```
Analyzing image: sample_images/candidate1.jpg
============================================================
Running head pose detection...
  ✓  Head pose: Forward
Running eye tracking...
  ✓  Gaze direction: CENTER
Running object detection...
  ⚠️  Object violation: Phone detected
  ✓  Object detection: Normal
     - Faces: 1
     - Hands: 2

============================================================
⚠️  ANALYSIS RESULT: SUSPICIOUS - 1 violation(s) detected
============================================================

Annotated image saved: results/candidate1_analyzed.jpg
JSON report saved: results/candidate1_report.json
```

## Troubleshooting

### ModuleNotFoundError
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Model Download Issues
If the EfficientDet model fails to download automatically:
1. Download manually from the URL in the error message
2. Place `efficientdet_lite0.tflite` in the image-proctoring directory

### Image Not Found
- Ensure the image path is correct
- Use absolute paths if relative paths don't work
- Check file extensions (.jpg, .jpeg, .png, .bmp)

## Limitations

1. **Static Analysis Only**: Cannot detect temporal behaviors (e.g., looking away briefly)
2. **No Audio**: Cannot detect conversations or suspicious sounds
3. **Single Frame**: Miss context that video would provide
4. **Lighting**: Requires well-lit images for accurate detection

## Use Cases

Perfect for:
- Post-exam image review
- Batch analysis of exam photos
- Automated screening before manual review
- Testing and validation of detection algorithms
- Processing archived exam images

## Comparison with Video Stream Version

**Video Stream Advantages:**
- Real-time monitoring
- Temporal violation detection
- Audio monitoring
- Continuous surveillance

**Image Analysis Advantages:**
- Process existing photos
- Batch processing capability
- No camera/microphone required
- Lower resource requirements
- Easier to archive and review
- Can be used offline

## Future Enhancements

- [ ] Support for analyzing frames extracted from video files
- [ ] Confidence scoring system
- [ ] ML-based violation severity ranking
- [ ] Web interface for bulk image upload
- [ ] Integration with exam management systems
- [ ] Support for additional object categories

## License

Same as the main proctoring application.

## Related

- Main proctoring application (video stream): `../proctoring_app.py`
- Rust version: `../src/main.rs`

---

For questions or issues, please refer to the main project documentation.
# image-proctoring
