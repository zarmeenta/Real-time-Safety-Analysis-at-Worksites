from ultralytics import YOLO
import torch
# Load YOLO model with pretrained weights

# Set your desired output path
output_path = "yolo_finetuning"  # Update this with your desired save location

def train_model():
    # Load the YOLOv11 model
    model = YOLO('yolo11n.pt')  # Replace 'yolov11.pt' with your YOLOv11 weight file if you have one

    # Set CUDA for GPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Train the model on the Open Images V7 dataset
    results = model.train(data=r"/home/vector/Desktop/CV_finetuning_YOLOV11/safety-Helmet-Reflective-Jacket/data.yaml", epochs=15, 
                        imgsz=640, amp=True, device=device, name='yolov11_helmets_vests', project=output_path)

    print("Training complete. Results saved at:", output_path)

    # Save the model
    model.export(format='torchscript', imgsz=640)
    # model.export(format='torchscript', path=r'C:\Users\dell\Desktop\Z\Semester 7\CV\Project\yolo11n_finetuned.pt')  # or choose a different format



if __name__ == '__main__':
    train_model()
