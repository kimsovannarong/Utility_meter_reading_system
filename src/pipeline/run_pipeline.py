# # In your main application script
# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# import cv2

# class MeterReadingPipeline:
#     def __init__(self):
#         # Load your best YOLO detector
#         self.detector = YOLO('01_detection/best_model.pt')
        
#         # Load your custom trained OCR
#         self.ocr = PaddleOCR(
#             rec_model_dir='PaddleOCR/inference/rec_digit/',  # Your trained model
#             use_angle_cls=False,
#             lang='en',
#             show_log=False
#         )
    
#     def process_image(self, image_path):
#         # Step 1: Detect meter and digit regions
#         results = self.detector(image_path)
        
#         # Step 2: Extract digit region crops
#         digit_crops = []
#         for result in results:
#             if result.boxes is not None:
#                 for box in result.boxes:
#                     if int(box.cls) == 1:  # Digit region class
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
#                         img = cv2.imread(image_path)
#                         crop = img[y1:y2, x1:x2]
#                         digit_crops.append(crop)
        
#         # Step 3: Read digits with custom OCR
#         readings = []
#         for crop in digit_crops:
#             # Save temp crop for OCR
#             cv2.imwrite('temp_crop.jpg', crop)
            
#             # Use your custom trained model
#             ocr_result = self.ocr.ocr('temp_crop.jpg', cls=False)
            
#             if ocr_result and ocr_result[0]:
#                 text = ocr_result[0][0][1][0]  # Extract recognized text
#                 readings.append(text)
        
#         return readings

# # Usage
# pipeline = MeterReadingPipeline()
# result = pipeline.process_image('test_meter.jpg')
# print(f"📊 Meter Reading: {result}")