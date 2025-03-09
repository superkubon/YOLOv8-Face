import os
import cv2
import torch
import argparse
from ultralytics import YOLO
from datetime import datetime

def parse_variables():
    parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights', default='yolov8m-face.pt')
    parser.add_argument('--conf', type=float, help='Confidence threshold', default=0.5)
    parser.add_argument('--output', type=str, help='Output folder for images', default='output')
    return vars(parser.parse_args())

def main():
    # Parse arguments
    variables = parse_variables()
    output_folder = variables['output']

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Select device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load YOLOv8 model
    model = YOLO(variables['weights']).to(device)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, conf=variables["conf"])

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
# Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                if conf > variables["conf"]:  # Only draw if confidence is high enough
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Add padding to prevent a tight crop
                    padding = 20  # Increase this if needed
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)

                    # Crop the expanded face area
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
                        image_path = os.path.join(output_folder, f"face_{timestamp}.jpg")
                        cv2.imwrite(image_path, face_crop)
                        print(f"📷 Saved face: {image_path}")


        # Show the frame
        cv2.imshow("YOLOv8 Face Detection", frame)

        # Press 's' to save the entire frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(output_folder, f"frame_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"📷 Manually saved: {image_path}")

        # Press 'q' to exit
        if key == ord('q'):
            break

    # Release webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
