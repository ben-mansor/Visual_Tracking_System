import cv2
from ultralytics import YOLOv10


model = YOLOv10("yolov10b.pt")  

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{classes[class_id]}: {confidence:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_frame(frame):
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls)
            confidence = float(box.conf)
            draw_bounding_box(frame, class_id, confidence, x1, y1, x2, y2)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("Object Detection and Tracking", frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # Press ESC to exit
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        processed_image = process_frame(image)
        cv2.imshow("Object Detection", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not read the image.")

if __name__ == "__main__":
    mode = input("Enter 'cam' for webcam, 'video' for video file, or 'image' for image file: ").strip().lower()
    if mode == 'cam':
        process_video(0)
    elif mode == 'video':
        video_path = input("Enter the path to the video file: ").strip()
        process_video(video_path)
    elif mode == 'image':
        image_path = input("Enter the path to the image file: ").strip()
        process_image(image_path)
    else:
        print("Invalid input. Please enter 'cam', 'video', or 'image'.")
