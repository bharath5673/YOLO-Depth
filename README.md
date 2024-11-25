## YOLO Object Detection with Intels'MiDaS Depth

This project integrates real-time object detection using YOLO models with depth estimation provided by Intel's MiDaS. 
It enables detection and localization of objects in 3D space, enhancing understanding beyond traditional 2D bounding boxes. 
Ideal for robotics, ADAS, and surveillance applications, it provides an intuitive top-down view of detected objects and their relative distances.

![demo](output_video_depth_output_pip_.gif)

### Features

- Real-time object detection using the YOLO models.
- Detection of various objects.
- Leverages Intel's MiDaS for depth information, transforming 2D detections into spatially aware 3D data.
- depth infos maps in real time.
- Customizable confidence threshold and class filtering.
- Simulated environment provides an intuitive top-down view of object positions and movements.
- Easy integration with pre-trained YOLO models.
- Provides bounding box coordinates, class labels, and tracking IDs for detected objects.

### Prerequisites

- Python 3.x
- OpenCV
- PyTorch
- NumPy
- Ultralytics

### Installation

1. Clone this repository.
2. Install the required dependencies

```bash
pip3 install ultralytics timm six
```

### Usage

1. Download pre-trained YOLOv5 weights or train your own model.
2. Provide the path to the YOLOv5 weights in the code.
3. Run the script with the video file.
4. View the object detection results and Bird's Eye View visualization.

For more detailed usage instructions and options, refer to the project documentation.

### Run

```bash
python3 main.py
```

### Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### References

* [Ultralytics YOLO](https://github.com/ultralytics)
* [Intel MiDaS](https://github.com/isl-org/MiDaS)


### License
