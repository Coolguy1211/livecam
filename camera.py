import cv2
import imutils
import numpy as np
import os
import threading
import time

class Camera:
    def __init__(self, url, camera_id=0):
        self.url = url
        self.camera_id = camera_id
        self.is_running = False
        self.thread = None
        self.output_frame = None
        self.lock = threading.Lock()

        # --- Load YOLO model ---
        yolo_dir = "yolo"
        yolo_config = os.path.join(yolo_dir, "yolov3.cfg")
        yolo_weights = os.path.join(yolo_dir, "yolov3.weights")
        yolo_classes_file = os.path.join(yolo_dir, "coco.names")

        print(f"[INFO] Camera {self.camera_id}: Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)

        with open(yolo_classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        try:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def start(self):
        """Starts the camera thread."""
        if self.is_running:
            print(f"[WARNING] Camera {self.camera_id}: Thread already running.")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._run, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"[INFO] Camera {self.camera_id}: Thread started.")

    def stop(self):
        """Stops the camera thread."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
        print(f"[INFO] Camera {self.camera_id}: Thread stopped.")

    def _run(self):
        """The main loop for the camera thread."""
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            print(f"[ERROR] Camera {self.camera_id}: Could not open video stream.")
            self.is_running = False
            return

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Camera {self.camera_id}: Could not read frame. Reconnecting...")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(self.url)
                continue

            frame = imutils.resize(frame, width=800)
            processed_frame = self._process_frame(frame)

            with self.lock:
                self.output_frame = processed_frame.copy()

        cap.release()

    def _process_frame(self, frame):
        """Performs object detection on a single frame."""
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.output_layers)

        boxes, confidences, classIDs = [], [], []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                # Draw bounding box
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # --- Notification Logic ---
                # Check if a person is detected and log it
                if self.classes[classIDs[i]] == "person":
                    self._log_notification("Person detected")

        return frame

    def _log_notification(self, message):
        """Logs a notification message to a file, with a cooldown."""
        log_file = "notifications.log"
        cooldown = 30  # seconds

        # Check if we need to initialize the last_log_time
        if not hasattr(self, 'last_log_time'):
            self.last_log_time = 0

        current_time = time.time()
        if (current_time - self.last_log_time) > cooldown:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] [Camera {self.camera_id}] {message}\n"

            with open(log_file, "a") as f:
                f.write(log_message)

            self.last_log_time = current_time

    def get_frame(self):
        """Returns the latest processed frame."""
        with self.lock:
            if self.output_frame is None:
                return None

            (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
            if not flag:
                return None

            return encodedImage.tobytes()
