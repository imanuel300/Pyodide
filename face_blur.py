import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

def blur_face(image, face):
    (x, y, w, h) = map(int, face)
    face_roi = image[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
    image[y:y+h, x:x+w] = blurred_face
    return image

def detect_faces(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    confirmed_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 1:
            aspect_ratio = float(w) / h
            relative_size = (w * h) / (img_array.shape[0] * img_array.shape[1])
            if 0.5 < aspect_ratio < 1.5 and 0.01 < relative_size < 0.25:
                confirmed_faces.append([x, y, w, h])
    
    print(f"פנים שזוהו: {confirmed_faces}")  # הדפסה לצורך דיבוג
    
    # ציור מסגרות סביב הפנים שזוהו (לצורך בדיקה)
    debug_img = img_array.copy()
    for (x, y, w, h) in confirmed_faces:
        cv2.rectangle(debug_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    
    # Save debug image
    Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)).save('detected_faces_debug.png')
    
    for face in confirmed_faces:
        img_array = blur_face(img_array, face)
    
    # Save result image
    Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)).save('blurred_faces_large.png')
    
    print(f"התמונה עם {len(confirmed_faces)} פנים מטושטשות נשמרה בשם 'blurred_faces_large.png'")
    return len(confirmed_faces)

# This function will be called from JavaScript
def process_image(image_data):
    img_array = np.array(Image.open(BytesIO(base64.b64decode(image_data))))
    return detect_faces(img_array)
