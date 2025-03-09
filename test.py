from ultralytics import YOLO
import torch
import cv2
import argparse

def parse_variables():
    parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='yolov8m_200e.pt')
    parser.add_argument('-t', '--threshold', type=float, help='Score threshold',
                        default=0.5)
    
    parser.add_argument('-i', '--input', nargs='+', help='Sample input image path',
                        default=['test_images/test_input.jpg','test_images/test_input_2.jpg','test_images/test_input_3.jpg'])

    parser.add_argument('-o', '--output', nargs='+', help='Sample output image path',
                        default=['test_images/test_output.jpg', 'test_images/test_output_2.jpg', 'test_images/test_output_3.jpg'],)


    
    args = parser.parse_args()
    variables = vars(args)
    return variables

def main():
    # Parse the command line arguments
    variables = parse_variables()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device

    # Ensure output folder exists
    output_dir = os.path.dirname(variables['output'][0])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the pretrained model
    model = YOLO(variables['weights']).to(device)

    # Read input images and check if they loaded correctly
    imgs = [cv2.imread(f) for f in variables['input']]
    imgs = [img for img in imgs if img is not None]  # Remove failed images

    if not imgs:
        print("❌ No valid input images found. Check file paths.")
        return

    # Run Predictions
    results = model.predict(imgs, verbose=True)  

    for i, result in enumerate(results):
        image = imgs[i]
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        if len(bboxes) == 0:
            print(f"⚠️ No faces detected in {variables['input'][i]}")

        for j in range(len(bboxes)):
            (x1, y1, x2, y2), score = bboxes[j], scores[j]

            if score >= variables["threshold"]:
                cv2.putText(image, f"Face {score:.4f}", (int(x1), int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Save image with detections
        saved = cv2.imwrite(variables['output'][i], image)
        if saved:
            print(f"✅ Image saved: {variables['output'][i]}")
        else:
            print(f"❌ Failed to save image: {variables['output'][i]}")

    

if __name__ == '__main__':
    main()
