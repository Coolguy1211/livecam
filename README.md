# Live AI Camera Monitoring Suite

This project provides a basic framework for a live AI camera monitoring suite. It can connect to any IP camera stream, with a specific focus on the "IP Webcam" Android app.

## Features

*   Connects to any IP camera stream.
*   Displays the live video feed in a window.
*   Performs real-time object detection using the YOLOv3 model.

## Prerequisites

*   Python 3.6+
*   An IP camera or an Android phone with the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app installed.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the camera URL:**
    *   Open the `main.py` file.
    *   Find the `URL` variable.
    *   Replace the placeholder URL with the stream URL of your IP camera.
    *   If you are using the "IP Webcam" app, start the server in the app and you will see a URL on your phone's screen. It will look something like `http://192.168.1.10:8080`. You need to add `/video` to the end of it, so the final URL will be `http://192.168.1.10:8080/video`.

## AI Model

This project uses the YOLOv3 model for object detection. You will need to download the pre-trained weights for the model to work.

**Download the model weights:**

1.  Download the `yolov3.weights` file from [this link](https://huggingface.co/prakhar5342/yolov3-model/resolve/main/yolov3.weights).
2.  Place the downloaded `yolov3.weights` file inside the `yolo` directory in the root of this project.

## Usage

Run the main script:
```bash
python main.py
```

A window should appear showing the live feed from your camera with object detection bounding boxes.
To stop the application, press the 'q' key while the video window is in focus.
