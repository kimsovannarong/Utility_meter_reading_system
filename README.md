Step to set up this project: 
1. Clone the project repositories via : https://github.com/kimsovannarong/Utility_meter_reading_system.git

2. Create a virtual environment : python -m venv your_environment_name

3. Activate the virtual environment : your_environment_name\Scripts\activate
 
4. Install all required dependencies : pip install -r requirements.txt

5. Creating hidden folder (i did not push): 
   models/ detectors (From parent folder)
   models/ ocr_model (From parent folder)
   experiments/detr  (From parent folder) 
   experiments/yolov8  (From parent fodler)
   experiments/yolov10 (From parent folder)
   experiments/yolov11 (From parent folder)

6. To train model (Activate the environment first): 
   1. python src/detectors/train/train_yolov8.py --epochs 25
   2. python src/detectors/train/train_yolov8.py --epochs 25 --batch 12 (Adjust batch size)

7. To evaluate model 
   1. python src/detectors/train/train_yolov8.py --evaluate 

8. To test the model (Require to input the model.pt path)
   1. python src/detectors/test/test_yolov8.py --model models\detectors\YOLOv8\yolov8_e25_b4\weights\best.pt>

9. Observe your result in console 
10. Input your metrics result in our Google docs   