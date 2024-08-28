# Employee Monitoring System

## Overview

This Employee Monitoring System is a video processing application that performs real-time analysis of employee behavior in a workplace setting. It uses computer vision and machine learning techniques to identify employees, verify dress code compliance, and detect greeting behaviors.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Main Thread                                 │
│                                                                         │
│  ┌───────────────────┐        ┌───────────────────┐                     │
│  │  Initialize       │        │ Create Queues &   │                     │
│  │  Embedding Index  │───────▶│  Start Threads    │                     │
│  └───────────────────┘        └───────────────────┘                     │
│                                   │                                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                 ┌──────────────────┴───────────────────┐
                 │                                      │
                 ▼                                      ▼
┌────────────────────────────────┐      ┌────────────────────────────────┐
│    Human Detection Thread      │      │  Face Verification, Dress &    │
│                                │      │      Greeting Thread           │
│  ┌───────────────────────────┐ │      │  ┌───────────────────────────┐ │
│  │ Initialize Model & Video  │ │      │  │   Get Frame from Queue    │ │
│  └───────────────────────────┘ │      │  └───────────────────────────┘ │
│              │                 │      │              │                 │
│              ▼                 │      │              ▼                 │
│  ┌───────────────────────────┐ │      │  ┌───────────────────────────┐ │
│  │    Process Video Frames   │ │      │  │      Process Boxes        │ │
│  └───────────────────────────┘ │      │  └───────────────────────────┘ │
│              │                 │      │              │                 │
│              ▼                 │      │              ▼                 │
│  ┌───────────────────────────┐ │      │  ┌───────────────────────────┐ │
│  │  Put Results in Queue     │ │      │  │    Face Verification      │ │
│  └───────────────────────────┘ │      │  └───────────────────────────┘ │
│              │                 │      │              │                 │
│              ▼                 │      │              ▼                 │
│  ┌───────────────────────────┐ │      │  ┌───────────────────────────┐ │
│  │   Plot Results on Frame   │ │      │  │    Dress Verification     │ │
│  └───────────────────────────┘ │      │  └───────────────────────────┘ │
│              │                 │      │              │                 │
│              ▼                 │      │              ▼                 │
│  ┌───────────────────────────┐ │      │  ┌───────────────────────────┐ │
│  │      Display Frame        │ │      │  │    Greeting Detection     │ │
│  └───────────────────────────┘ │      │  └───────────────────────────┘ │
│                                │      │              │                 │
└────────────────────────────────┘      │              ▼                 │
               ▲                        │  ┌───────────────────────────┐ │
               │                        │  │  Update Identify Queue    │ │
               │                        │  └───────────────────────────┘ │
               │                        └────────────────────────────────┘
               │                                       │
               │                                       │
               └───────────────────────────────────────┘
```

## Features

- Human detection in video streams
- Face verification against a database of known employees
- Dress code compliance verification
- Greeting behavior detection
- Real-time tracking of individuals across video frames
- Event logging with timestamps
- Live visualization of processed video frames

## Prerequisites

- Python 3.7+
- OpenCV
- PyTorch
- Ultralytics YOLOv8

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/vTuanpham/Employee_monitor.git
   cd Employee_monitor
   ```

2. Install the required packages:
   ```
   bash setup.sh
   ```

## Usage

Run the main script:

```
python monitor_pipe/pipe.py
```

By default, the script will use the video file specified in `VIDEO_FILE_1`. To use a different video file, modify the `VIDEO_FILE_1` constant in the script.
To use Webcam, change `VIDEO_FILE_1` to 0

## Configuration

The following constants can be adjusted in the script:

- `VIDEO_FILE_1`: Path to the input video file, can be set to 0 for Webcam
- `MODEL_FILE`: Path to the YOLOv8 model file
- `DEVICE_ID`: GPU device ID 
- `confidence_threshold`: Threshold for face verification confidence (default: 0.65)
- `max_age_seconds`: Maximum age of a track ID in seconds (default: 10)
- `cooldown_seconds`: Cooldown period for greeting detection in seconds (default: 5)

## Logging

The system logs important events to `identification_log.txt`. This includes:

- Employee identifications
- Dress code compliance
- Greeting detections

## Disclaimer

This system is designed for educational and research purposes. Ensure compliance with all applicable laws and regulations regarding workplace monitoring and data privacy before deploying in a real-world setting.
