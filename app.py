from flask import Flask, Response, render_template
from camera import Camera
import time
import threading
import os
import cv2
import configparser

# --- Configuration ---
config = configparser.ConfigParser()
config.read('config.ini')

CAMERA_URLS = config['DEFAULT']['camera_urls'].split(',')
YOLO_CONFIG = config['DEFAULT']['yolo_config']
YOLO_WEIGHTS = config['DEFAULT']['yolo_weights']
YOLO_CLASSES = config['DEFAULT']['yolo_classes']


# --- Global Variables ---
cameras = []
app = Flask(__name__)
net = None
output_layers = None
classes = None

def load_yolo_model():
    """Loads the YOLO model from disk."""
    global net, output_layers, classes
    print("[INFO] Loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)

    with open(YOLO_CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def start_cameras():
    """Initializes and starts all camera threads."""
    global net, output_layers, classes, config
    if net is None:
        load_yolo_model()

    for i, url in enumerate(CAMERA_URLS):
        camera = Camera(url, net, output_layers, classes, config, camera_id=i)
        cameras.append(camera)
        camera.start()
    print(f"[INFO] Started {len(cameras)} camera threads.")

@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html", num_cameras=len(CAMERA_URLS))

def gen_frame(camera_id):
    """Generator function for video streaming."""
    camera = cameras[camera_id]
    while True:
        frame_bytes = camera.get_frame()
        if frame_bytes is None:
            # You can send a placeholder image here if a frame is not available
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if camera_id >= len(cameras):
        return "Camera not found", 404
    return Response(gen_frame(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the camera threads in the background
    camera_thread = threading.Thread(target=start_cameras)
    camera_thread.daemon = True
    camera_thread.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)

    # Cleanup: stop camera threads when the app is shut down
    print("[INFO] Stopping all cameras...")
    for camera in cameras:
        camera.stop()
    print("[INFO] All cameras stopped.")
