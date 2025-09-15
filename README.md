# Live AI Camera Monitoring Suite

This project is a web-based AI camera monitoring suite that can connect to multiple IP camera streams, perform real-time object detection, and provide notifications.

## Features

*   **Web-Based Interface:** Access your camera feeds from anywhere with a web browser.
*   **Multi-Camera Support:** Monitor multiple cameras simultaneously.
*   **Real-Time AI Detection:** Uses the YOLOv3 model to detect 80 different types of objects.
*   **Notification System:** Logs a notification to `notifications.log` when a person is detected.
*   **Documentation:** A documentation website is available via GitHub Pages.

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

3.  **Download the AI Model Weights:**
    *   This project uses the YOLOv3 model. You need to download the pre-trained weights file.
    *   Download `yolov3.weights` from [this link](https://huggingface.co/prakhar5342/yolov3-model/resolve/main/yolov3.weights).
    *   Place the downloaded `yolov3.weights` file inside the `yolo` directory.

4.  **Configure Camera URLs:**
    *   Open the `app.py` file.
    *   Find the `CAMERA_URLS` list.
    *   Replace the placeholder URLs with the stream URLs of your IP cameras. You can add as many URLs as you need.
    *   For the "IP Webcam" app, the URL is typically `http://<your-phone-ip>:8080/video`.

## Usage

1.  **Run the web server:**
    ```bash
    python app.py
    ```

2.  **Access the web interface:**
    *   Open your web browser and navigate to `http://127.0.0.1:5000`.
    *   You should see the video feeds from all your configured cameras.

## Notifications

*   When a "person" is detected on any camera feed, a notification is logged.
*   The notifications are saved in the `notifications.log` file in the root of the project.
*   A cooldown of 30 seconds is applied to prevent spamming the log for the same camera.
