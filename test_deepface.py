import cv2
from ultralytics import YOLO
import numpy as np
import os
import pickle
import time



DB_FILE = "face_database.pkl"
RECOGNITION_THRESHOLD = 80  # Lower = more strict matching
YOLO_CONFIDENCE = 0.5
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

model = YOLO("yolov8n-face.pt")


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_database = {}  
face_id_counter = 0
face_samples = []    
face_labels = []     


def load_database():
    global face_database, face_id_counter, face_recognizer, face_samples, face_labels
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'rb') as f:
                data = pickle.load(f)
                face_database = data['mapping']
                face_id_counter = data['counter']
                if os.path.exists("face_recognizer_model.xml"):
                    face_recognizer.read("face_recognizer_model.xml")
                print(f"Loaded {len(face_database)} known faces from database")
        except:
            print("Error loading database, starting fresh")
            face_database = {}
            face_id_counter = 0
    else:
        print("No existing database found")

def save_database():
    data = {
        'mapping': face_database,
        'counter': face_id_counter
    }
    with open(DB_FILE, 'wb') as f:
        pickle.dump(data, f)
    face_recognizer.write("face_recognizer_model.xml")
    print(f"Saved {len(face_database)} faces to database")

def train_recognizer():
    """Train the face recognizer with current samples"""
    if len(face_samples) > 0:
        print(f"Training recognizer with {len(face_samples)} samples...")
        face_recognizer.train(face_samples, np.array(face_labels))
        save_database()

# =============================================================================
# FACE PROCESSING FUNCTIONS
# =============================================================================
def preprocess_face(face_img):
    """Convert face image to grayscale and equalize histogram"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (100, 100))
    return gray

def add_face_to_database(name, face_img):
    global face_id_counter, face_samples, face_labels
    
    if name not in face_database:
        face_id_counter += 1
        face_database[name] = face_id_counter
    
    face_id = face_database[name]
    processed_face = preprocess_face(face_img)
    
    face_samples.append(processed_face)
    face_labels.append(face_id)
    
    train_recognizer()
    print(f"Added {name} to database (ID: {face_id})")

def recognize_face(face_img):
    """Recognize face using LBPH recognizer"""
    if len(face_database) == 0:
        return "Unknown", 0
    
    processed_face = preprocess_face(face_img)
    
    try:
        label, confidence = face_recognizer.predict(processed_face)
        
        for name, id_val in face_database.items():
            if id_val == label:
                return name, confidence
        
        return "Unknown", confidence
    except:
        return "Unknown", 0


def main():
    load_database()
    
    print("=" * 50)
    print("FACE RECOGNITION SYSTEM")
    print("=" * 50)
    print("Controls:")
    print("- Press 'a' to add the currently detected face to database")
    print("- Press 'd' to display database contents")
    print("- Press 'c' to clear all faces from database")
    print("- Press 'q' to quit")
    print("=" * 50)
    
    # For smoothing recognition results
    recognition_history = {}
    smoothing_window = 5
    
    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to capture video")
            break
        
        # Run YOLO face detection
        results = model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)
        annotated_frame = results[0].plot()
        
        # Extract bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        
        # Process each detected face
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Add padding
            padding = 20
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(frame.shape[1], x2 + padding)
            y2_padded = min(frame.shape[0], y2 + padding)
            
            face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if face_region.size == 0:
                continue
            
            # Get a unique ID for this face position (for smoothing)
            face_id = f"{x1}_{y1}"
            
            # Only process recognition every few frames to improve performance
            current_time = time.time()
            if face_id not in recognition_history:
                recognition_history[face_id] = {'last_recognition': 0, 'name': "Unknown", 'confidence': 0}
            
            if current_time - recognition_history[face_id]['last_recognition'] > 0.5:  # Process every 0.5 seconds
                name, confidence = recognize_face(face_region)
                recognition_history[face_id] = {
                    'last_recognition': current_time,
                    'name': name,
                    'confidence': confidence
                }
            
            # Use the smoothed recognition result
            name = recognition_history[face_id]['name']
            confidence = recognition_history[face_id]['confidence']
            
            # Draw recognition result
            label = f"{name} ({confidence:.1f})"
            color = (0, 255, 0) if name != "Unknown" and confidence < RECOGNITION_THRESHOLD else (0, 0, 255)
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display info
        cv2.putText(annotated_frame, "Press 'a' to add face, 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Known faces: {len(face_database)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition System", annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a') and boxes is not None and len(boxes) > 0:

            x1, y1, x2, y2 = map(int, boxes[0][:4])
            padding = 20
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(frame.shape[1], x2 + padding)
            y2_padded = min(frame.shape[0], y2 + padding)
            
            face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if face_region.size > 0:
                cv2.imshow("Face to Add", face_region)
                cv2.waitKey(100)
                
                name = input("Enter name for this face: ").strip()
                if name:
                    add_face_to_database(name, face_region)
                cv2.destroyWindow("Face to Add")
        elif key == ord('d'):
            print("\nCurrent database contents:")
            for i, name in enumerate(face_database.keys(), 1):
                print(f"{i}. {name}")
            print()
        elif key == ord('c'):
            confirm = input("Are you sure you want to clear the database? (y/n): ")
            if confirm.lower() == 'y':
                global face_samples, face_labels
                face_database.clear()
                face_samples = []
                face_labels = []
                face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                if os.path.exists("face_recognizer_model.xml"):
                    os.remove("face_recognizer_model.xml")
                save_database()
                print("Database cleared!")
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()