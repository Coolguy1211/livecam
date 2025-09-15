import cv2
import imutils
import numpy as np
import os
import threading
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class Camera:
    def __init__(self, url, net, output_layers, classes, config, camera_id=0):
        self.url = url
        self.camera_id = camera_id
        self.is_running = False
        self.thread = None
        self.output_frame = None
        self.lock = threading.Lock()
        self.net = net
        self.output_layers = output_layers
        self.classes = classes
        self.config = config

        # --- Notification Settings ---
        self.last_notification_time = 0
        self.notification_cooldown = 30  # seconds

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
                # Check if a person is detected and trigger a notification
                if self.classes[classIDs[i]] == "person":
                    self._trigger_notification("Person detected")

        return frame

    def _trigger_notification(self, message):
        """Triggers all configured notification methods, with a cooldown."""
        current_time = time.time()
        if (current_time - self.last_notification_time) > self.notification_cooldown:
            self.last_notification_time = current_time

            # Log to file
            self._log_to_file(message)

            # Send email
            if self.config.getboolean('EMAIL', 'enable'):
                self._send_email_notification(message)

    def _log_to_file(self, message):
        """Logs a notification message to a file."""
        log_file = "notifications.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [Camera {self.camera_id}] {message}\n"
        with open(log_file, "a") as f:
            f.write(log_message)

    def _send_email_notification(self, message):
        """Sends an email notification."""
        try:
            smtp_server = self.config['EMAIL']['smtp_server']
            smtp_port = self.config.getint('EMAIL', 'smtp_port')
            smtp_username = self.config['EMAIL']['smtp_username']
            smtp_password = self.config['EMAIL']['smtp_password']
            recipient_email = self.config['EMAIL']['recipient_email']

            msg = MIMEMultipart()
            msg['From'] = smtp_username
            msg['To'] = recipient_email
            msg['Subject'] = f"Person Detected on Camera {self.camera_id}"

            body = f"A person was detected on camera {self.camera_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_username, recipient_email, text)
            server.quit()
            print(f"[INFO] Camera {self.camera_id}: Email notification sent to {recipient_email}")

        except Exception as e:
            print(f"[ERROR] Camera {self.camera_id}: Failed to send email notification: {e}")

    def get_frame(self):
        """Returns the latest processed frame."""
        with self.lock:
            if self.output_frame is None:
                return None

            (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
            if not flag:
                return None

            return encodedImage.tobytes()
