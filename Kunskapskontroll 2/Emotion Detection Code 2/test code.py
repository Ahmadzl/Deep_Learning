#Import library
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model_fite.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#  Mapping label indices to emotions
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Open the video capture device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the face ROI to match the input size of the model (48x48)
        resized_face = cv2.resize(face_roi, (48, 48)) / 255.0
        
        # Reshape the resized face for model prediction
        reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))
        
        # Make prediction using the trained model
        result = model.predict(reshaped_face)
        
        # Get the index of the predicted label
        label_index = np.argmax(result)
        
        # Get the corresponding emotion label
        emotion_label = labels_dict[label_index]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the predicted emotion label above the rectangle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()
