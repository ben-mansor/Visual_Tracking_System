import cv2
import numpy as np

#YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Load names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the video capture object
# Use 0 for the webcam or put vid path instead
cap = cv2.VideoCapture(0)  

# output layer names in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Start processing 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detection data
    class_ids = []
    confidences = []
    boxes = []

    # Loop through each of the layer outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the detected objects
    if len(indices) > 0:
        for i in indices:
            if isinstance(i, (list, np.ndarray)):
                i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            draw_bounding_box(frame, class_ids[i], confidences[i], x, y, x + w, y + h)

    # Display the frame with the detected objects
    cv2.imshow("Object Detection and Tracking", frame)

    # Exit
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
