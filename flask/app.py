from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned and pretrained YOLO models
finetuned_model_path = r"best.pt"
finetuned_model = YOLO(finetuned_model_path)
pretrained_model = YOLO('yolo11n.pt')

# Image preprocessing
def preprocess_image(image_input):
    return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

# Function to draw bounding boxes and calculate safety scores
def draw_boxes(results, image, class_names=None, custom_labels=None, safety_scores=None):
    safety_scores_idx = 0  # Initialize an index to track safety scores for persons
    
    img_width = image.shape[1]  # Image width for boundary checks
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates (x1, y1, x2, y2)
        confs = result.boxes.conf.cpu().numpy()  # Get confidence scores
        cls_ids = result.boxes.cls.cpu().numpy()  # Get class indices

        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            x1, y1, x2, y2 = map(int, box)
            if custom_labels:  # Handle custom class names (e.g., for pretrained model)
                class_name = custom_labels[cls_id]
            elif class_names:
                class_name = class_names[int(cls_id)]
            else:
                class_name = f"Class {int(cls_id)}"
            
            label = f"{class_name}: {conf:.2f}"

            # Apply confidence threshold checks here
            if (class_name == "person" and conf < 0.3) or \
               (class_name == "Safety-Helmet" and conf < 0.4) or \
               (class_name == "Reflective-Jackets" and conf < 0.3) or \
               (class_name == "cell phone" and conf < 0.4):
                continue  # Skip drawing this box if the confidence is too low

            if class_name == "person":
                # Ensure there's a valid safety score for this person
                if safety_scores and safety_scores_idx < len(safety_scores):
                    safety_score = safety_scores[safety_scores_idx]
                    safety_scores_idx += 1
                    label = f"Safety Score: {safety_score:.2f}"

                    # Set the bounding box color based on safety score
                    if safety_score >= 90:
                        color = (0, 255, 0)  # Green
                    elif safety_score > 50:
                        color = (255, 255, 0)  # Yellow
                    else:
                        color = (255, 0, 0)  # Red

                # Drawing the box and label with appropriate color
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Adjust the position of the safety score label if it goes off-screen
                label_y_position = y1 - 10
                if label_y_position < 10:  # If the label is going off the top, move it down
                    label_y_position = y1 + 10
                
                # Adjust the position if the label goes off the left or right side
                label_x_position = x1
                label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                
                if label_x_position + label_width > img_width:  # If the label goes beyond the right side
                    label_x_position = x2 - label_width  # Move the label to the right side of the box
                
                if label_x_position < 0:  # If the label goes beyond the left side
                    label_x_position = 0  # Move the label to the leftmost side

                # Draw the safety score label at the adjusted position
                cv2.putText(image, label, (label_x_position, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Function to calculate overlap between two bounding boxes
def calculate_overlap(box1, box2):
    # Calculate intersection area
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union area
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)

    union_area = area1 + area2 - inter_area

    # IoU calculation
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def is_in_top_percentage(person_box, box, percentage):
    _, person_y1, _, person_y2 = person_box
    _, y1, _, y2 = box

    # Calculate the top region boundary
    top_boundary = person_y1 + (percentage / 100) * (person_y2 - person_y1)

    # Check if there is overlap between the helmet box and the top region
    return y1 < top_boundary and y2 > person_y1

# Calculate safety score
def calculate_safety_score(person_box, detections, iou_threshold=0):
    safety_score = 0
    for det_box, class_name, _ in detections:  # Unpack the confidence score but ignore it
        iou = calculate_overlap(person_box, det_box)
        # print(f"IOU for {class_name} with person: {iou}")  # Log IoU for debugging
        if class_name in ["cell phone", "Safety-Helmet", "Reflective-Jackets"]:
            if iou > iou_threshold:
                # print(f"Overlap detected with {class_name} - IOU: {iou}")
                if class_name == "cell phone":
                    # print("ss-30")
                    safety_score -= 30
                elif class_name == "Reflective-Jackets" and is_in_top_percentage(person_box, det_box, 80):
                    safety_score += 50
                    # print("ss+50")
                elif class_name == "Safety-Helmet" and is_in_top_percentage(person_box, det_box, 70):
                    safety_score += 50
                    # print("ss+50")
    return safety_score


def dual_inference(image_input):
    """
    Perform dual model inference on the input image, which can be an image array or a file path.
    """
    image = preprocess_image(image_input)
    
    # Perform predictions using both models
    finetuned_results = finetuned_model.predict(source=image, save=False)
    pretrained_results = pretrained_model.predict(source=image, save=False, classes=[0, 67])

    finetuned_classes = ['Safety-Helmet', 'Reflective-Jackets']
    pretrained_labels = {0: "person", 67: "cell phone"}
    
    # Gather detections from the fine-tuned model
    detections = [
        (box.xyxy.cpu().numpy().flatten(), finetuned_classes[int(box.cls.cpu().numpy().item())], box.conf.cpu().item())
        for result in finetuned_results
        for box in result.boxes
    ]
    
    # Gather detections from the pretrained model (both person and cell phone)
    pretrained_detections = [
        (box.xyxy.cpu().numpy().flatten(), pretrained_labels[int(box.cls.cpu().numpy().item())], box.conf.cpu().item())
        for result in pretrained_results
        for box in result.boxes if int(box.cls.cpu().numpy().item()) in pretrained_labels
    ]
    
    # Combine detections from both models
    detections.extend(pretrained_detections)

    safety_scores = []
    # Calculate safety score for each detected person
    for person_box, class_name, confidence in pretrained_detections:
        if class_name == "person" and confidence >= 0.3:  # Only calculate for persons with sufficient confidence
            safety_score = calculate_safety_score(person_box, detections)
            safety_scores.append(safety_score)

    # Calculate the mean safety score for the entire scene (if there are any person safety scores)
    if safety_scores:
        scene_safety_score = sum(safety_scores) / len(safety_scores)
    else:
        scene_safety_score = 0.0

    # Set the border color based on the scene safety score
    if scene_safety_score >= 90:
        border_color = (0, 255, 0)  # Green
        safety_label_color = (0, 255, 0)
    elif scene_safety_score > 50:
        border_color = (255, 255, 0)  # Yellow
        safety_label_color = (255, 255, 0)
    else:
        border_color = (255, 0, 0)  # Red
        safety_label_color = (255, 0, 0)

    # Add border to the image
    image_with_border = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

    # Draw boxes and safety score on the image
    draw_boxes(finetuned_results, image_with_border, class_names=finetuned_classes, safety_scores=safety_scores)
    draw_boxes(pretrained_results, image_with_border, custom_labels=pretrained_labels, safety_scores=safety_scores)
    
    # Display scene safety score in the top-right corner
    text = f"Scene Safety Score: {scene_safety_score:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]  # Set the font scale to 0.4
    text_width = text_size[0]
    text_height = text_size[1]
    
    # Calculate the position of the text to ensure it stays in the top-right corner
    x_pos = image_with_border.shape[1] - text_width - 10  # 10px from the right edge
    y_pos = 30  # Fixed position at the top of the image
    
    # Ensure the text stays within the top-right corner
    if x_pos < 0:
        x_pos = image_with_border.shape[1] - text_width - 20  # Adjust left if text overflows
    
    cv2.putText(image_with_border, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, safety_label_color, 2)

    rgb_img = cv2.cvtColor(image_with_border, cv2.COLOR_BGR2RGB)

    return rgb_img  # Only return the processed image

# Generate frames for Flask video feed
def generate_frames():
    camera = cv2.VideoCapture(0)  # Access the webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform YOLO inference and process the frame
            processed_frame = dual_inference(frame)

            # Encode the processed frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route to stream video
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
