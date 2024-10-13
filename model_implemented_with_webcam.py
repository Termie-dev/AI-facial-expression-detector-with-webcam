import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize the webcam
webcam = cv2.VideoCapture(0)

model = load_model('/Users/tomassalian/Desktop/Developer/Webcam Emotion Detector/Executable Files/72_emotion_model.keras')

model.summary()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = webcam.read()

    if ret: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a green rectangle around each face
            
             # Extract the face ROI in color
            roi_color = frame[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (64, 64))
            roi_color = roi_color.astype('float32') / 255.0  # Normalize pixel values
            roi_color = img_to_array(roi_color)
            roi_color = np.expand_dims(roi_color, axis=0)

            # Predict the emotion
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
            predictions = model.predict(roi_color)[0]
            top_indices = np.argsort(-predictions)[:3]  # Get indices of top three predictions
            top_emotions = [(emotions[i], predictions[i]) for i in top_indices]

            for idx, (emotion, probability) in enumerate(top_emotions):
                emotionAndProbability = f"{emotion}: {probability:.2f}"
                cv2.putText(frame, emotionAndProbability, (x, y - 10 - (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        cv2.imshow('Emotion Detection Webcam', frame)  

        key = cv2.waitKey(1 ) 
        if key == ord('e'):  
            break

webcam.release()
cv2.destroyAllWindows()
