from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class TileDefectDetector:
    def __init__(self, checkpoint_path):
        """Initialize the YOLOv8 model"""
        self.model = YOLO(checkpoint_path)  # Load Ultralytics YOLOv8 model
        self.conf_thres = 0.3  # Confidence threshold (matching your example)
        # Define defect types and severity mapping
        self.defect_types = [
            'Broken tiles', 'Surface crack', 'Pin Hole', 'Scratch', 'Corner',
            'Corner chip', 'Edge chipping', 'Spot defect', 'Tiles-defects', 'Other'
        ]
        self.severity_map = {
            0: 'minor', 1: 'critical', 2: 'moderate', 3: 'critical', 4: 'moderate',
            5: 'moderate', 6: 'moderate', 7: 'minor', 8: 'critical', 9: 'minor'
        }

    def detect_defects(self, image):
        """Detect defects using YOLOv8"""
        try:
            print("Running model inference...")
            results = self.model.predict(image, conf=self.conf_thres)  # No save=True for Streamlit
            print("Processing predictions...")
            defects = self.process_predictions(results)
            print("Defect detection complete")
            return defects
        except Exception as e:
            raise Exception(f"Error detecting defects: {str(e)}")

    def process_predictions(self, results):
        """Convert YOLOv8 predictions to defect data"""
        defects = []
        for result in results:
            boxes = result.boxes
            orig_w, orig_h = result.orig_shape[1], result.orig_shape[0]  # Original image dimensions
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())  # Coordinates in original image space
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                defect = {
                    'defect_type': self.defect_types[class_id % len(self.defect_types)],
                    'severity': self.severity_map[class_id % len(self.severity_map)],
                    'x_position': (x1 / orig_w) * 100,  # Percentage of width
                    'y_position': (y1 / orig_h) * 100,  # Percentage of height
                    'width': ((x2 - x1) / orig_w) * 100,
                    'height': ((y2 - y1) / orig_h) * 100,
                    'confidence': confidence
                }
                print(f"Defect: {defect}")
                defects.append(defect)
        return defects

    def annotate_image(self, image, defects):
        """Draw defect boxes on the image, matching the YOLOv8 example"""
        print("Starting annotation...")
        img = image.copy()
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # Use default PIL font
        width, height = img.size
        
        for defect in defects:
            x1 = (defect['x_position'] * width) / 100
            y1 = (defect['y_position'] * height) / 100
            x2 = x1 + (defect['width'] * width) / 100
            y2 = y1 + (defect['height'] * height) / 100
            print(f"Drawing box at x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Draw bounding box (green, width=5 to match your example)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
            
            # Add label and confidence
            label = f"{defect['defect_type']} {defect['confidence']:.2f}"
            draw.text((x1, y1 - 10), label, fill="green", font=font)
        
        print("Annotation complete")
        return img
