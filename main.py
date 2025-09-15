import cv2
import imutils
import numpy as np
import os

# --- Constants ---
# TODO: Replace the URL with your IP camera's stream URL.
# For the IP Webcam app, this is usually http://<your-phone-ip>:8080/video
URL = "http://192.168.1.10:8080/video"

YOLO_DIR = "yolo"
YOLO_CONFIG = os.path.join(YOLO_DIR, "yolov3.cfg")
YOLO_WEIGHTS = os.path.join(YOLO_DIR, "yolov3.weights")
YOLO_CLASSES = os.path.join(YOLO_DIR, "coco.names")

def main():
    """
    Connects to an IP camera stream, performs object detection,
    displays the feed, and waits for the user to quit.
    """
    # --- Load YOLO model ---
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)

    # Load class names
    with open(YOLO_CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Get the output layer names
    layer_names = net.getLayerNames()
    # In OpenCV 4.x, getUnconnectedOutLayers() returns a 2D array, so we flatten it.
    # For older versions, it might return a 1D array, so handle that case.
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


    # --- Connect to camera ---
    cap = cv2.VideoCapture(URL)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Connecting to camera...")

    # Loop to continuously fetch frames from the stream
    while True:
        # Read a frame from the stream
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error: Could not read frame from stream. Exiting.")
            break

        # Resize the frame for better performance
        frame = imutils.resize(frame, width=800)
        (H, W) = frame.shape[:2]

        # --- AI Processing ---
        # Create a blob from the image and perform a forward pass of the
        # YOLO object detector
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(output_layers)

        # Initialize our lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []

        # Loop over each of the layer outputs
        for output in layerOutputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # Scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # --- Draw bounding boxes ---
        # Ensure at least one detection exists
        if len(idxs) > 0:
            # Loop over the indexes we are keeping
            for i in idxs.flatten():
                # Extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = (0, 255, 0) # Green
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        # Display the frame
        cv2.imshow("Live Camera Feed", frame)

        # Wait for the 'q' key to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
