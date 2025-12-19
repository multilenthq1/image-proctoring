# Server Logging Documentation

## Overview
The proctoring server now includes comprehensive logging functionality that tracks all analysis operations, violations, and server statistics.

## Log Directory Structure

```
server_logs/
├── server.log              # Main server log (rotating)
├── server.log.1            # Backup log files
├── server.log.2
└── analysis_log_YYYYMMDD_HHMMSS.json  # Analysis session logs
```

## Log Files

### 1. Server Log (`server.log`)
Main server activity log with automatic rotation (max 10MB per file, 5 backups).

**Contents:**
- Server initialization and shutdown
- Frame analysis requests
- Violation alerts
- Error messages
- Statistics summaries

**Example:**
```
2025-12-19 10:30:45 - proctoring_server - INFO - Initializing Proctoring Server...
2025-12-19 10:30:48 - proctoring_server - INFO - Server initialization complete!
2025-12-19 10:31:00 - proctoring_server - INFO - Analyzing frame 1 from client client_20251219_103100
2025-12-19 10:31:01 - proctoring_server - WARNING - Frame 1: SUSPICIOUS - 2 violations
2025-12-19 10:31:01 - proctoring_server - WARNING -   - Head pose violation: Looking Left
2025-12-19 10:31:01 - proctoring_server - WARNING -   - Eye tracking violation: Gaze Left
```

### 2. Analysis Log (`analysis_log_*.json`)
Detailed JSON logs of all frame analyses, automatically saved every 10 frames or when violations are detected.

**Structure:**
```json
{
  "session_start": "2025-12-19 10:30:45",
  "total_frames": 15,
  "suspicious_frames": 3,
  "normal_frames": 12,
  "analysis_log": [
    {
      "frame_id": 1,
      "timestamp": "2025-12-19 10:31:00",
      "is_suspicious": true,
      "violations": [
        "Head pose violation: Looking Left",
        "Eye tracking violation: Gaze Left"
      ],
      "client_id": "client_20251219_103100"
    }
  ]
}
```

## Server Statistics

### Real-time Statistics Endpoint
Get current server statistics via HTTP:

```bash
curl http://localhost:5000/stats
```

**Response:**
```json
{
  "session_start": "2025-12-19 10:30:45",
  "uptime_seconds": 1234.5,
  "total_frames_analyzed": 150,
  "suspicious_frames": 23,
  "normal_frames": 127,
  "suspicious_rate": 15.3
}
```

## Logging Levels

- **INFO**: Normal operations (frame analysis, server status)
- **WARNING**: Violations detected, suspicious frames
- **ERROR**: Analysis failures, decoding errors

## Client ID Tracking

Clients can now provide a unique identifier for tracking:

**Client:**
```bash
python client_stream.py --client-id exam_room_1
```

**Server logs will include:**
```
Analyzing frame 1 from client exam_room_1
```

This allows tracking multiple concurrent exam sessions.

## Log Rotation

Server logs automatically rotate when reaching 10MB:
- `server.log` (current)
- `server.log.1` (previous)
- `server.log.2` (older)
- ... up to `server.log.5`

Oldest logs are automatically deleted.

## Monitoring and Analysis

### View Recent Violations
```bash
tail -f server_logs/server.log | grep WARNING
```

### Count Suspicious Frames
```bash
grep "SUSPICIOUS" server_logs/server.log | wc -l
```

### Parse Analysis Logs
```python
import json

with open('server_logs/analysis_log_20251219_103045.json', 'r') as f:
    data = json.load(f)
    
print(f"Suspicious rate: {data['suspicious_frames'] / data['total_frames'] * 100:.1f}%")

# Get unique violations
violations = set()
for entry in data['analysis_log']:
    violations.update(entry['violations'])
print("Unique violations:", violations)
```

## Usage Tips

1. **Multiple Clients**: Use `--client-id` to track different exam rooms or students
2. **Log Analysis**: Parse JSON logs for detailed analytics and reporting
3. **Real-time Monitoring**: Monitor `server.log` for live violation alerts
4. **Statistics**: Check `/stats` endpoint for session overview
5. **Cleanup**: Old analysis logs can be archived or deleted as needed

## Example: Multi-Client Setup

**Server:**
```bash
python server.py
```

**Client 1 (Exam Room A):**
```bash
python client_stream.py --client-id room_a --source 0
```

**Client 2 (Exam Room B):**
```bash
python client_stream.py --client-id room_b --source 1
```

All frames from both clients are logged separately and can be tracked by `client_id` in the analysis logs.

## Log File Locations

- **Server logs**: `server_logs/` directory (created automatically)
- **Client logs**: `logs/` directory (created by client)

Both directories are created automatically on first use.
