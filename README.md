# YOLOv10 Object Detection and Tracking

This repository contains a Python script that implements real-time object detection and tracking using the YOLOv10 algorithm and OpenCV. The script captures video from a webcam, processes each frame using a YOLOv10 model, and displays the detected objects with bounding boxes.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x
- `opencv-python` and `opencv-python-headless`
- `numpy`
- YOLOv10 weights file (e.g., `yolov10x.pt`)
- COCO names file (e.g., `coco.names`)

### Install Dependencies

Use `pip` to install the required packages:
```bash
pip install opencv-python opencv-python-headless numpy
```

## Usage  

- Clone the repository:
  ```bash
  git clone https://github.com/ben-mansor/yolov10-object-detection.git
  cd yolov10-object-detection
  ```
- Ensure you have the YOLOv10 weights file (yolov10x.pt) and COCO names file (coco.names) in the same directory as the script or adjust the paths accordingly.
- Run the script:
  ```bash
  python yolov10_object_detection.py
  ```

## Contributing

### Contributions are welcome! Please follow these steps to contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit them (git commit -m 'Add feature').
- Push to the branch (git push origin feature-branch).
- Open a Pull Request.


