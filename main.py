import cv2
import numpy as np


# Load pre-trained model and configuration for person detection
net = cv2.dnn.readNetFromCaffe(r'model\fullface_deploy.prototxt', r'model\fullfacedetection.caffemodel')

# Open a video file or capture from camera
cap = cv2.VideoCapture(r'input\drum.mp4')  # Replace with '0' for default camera

# Set output video size according to the frame.
size = (600, 600)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)


while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to have a maximum width of 600 pixels
    frame = cv2.resize(frame, (600, 600))

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Prepare the frame for object detection by normalizing and resizing
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the pre-trained model
    net.setInput(blob)

    # Run forward pass to get predictions
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by confidence
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    # Display the frame
    cv2.imshow("Person Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
