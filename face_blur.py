import cv2
import numpy as np
from PIL import Image

def blur_face(image, face):
    (x, y, w, h) = map(int, face)
    face_roi = image[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
    image[y:y+h, x:x+w] = blurred_face
    return image

def detect_faces(img):
    # Convert to BGR for OpenCV if needed
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = img
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # זיהוי פנים עם פרמטרים מותאמים
    faces1 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    faces2 = face_cascade_alt.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    faces3 = face_cascade_alt2.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # איחוד כל הזיהויים
    all_faces = np.vstack([faces1, faces2, faces3]) if len(faces1) and len(faces2) and len(faces3) else \
                np.vstack([faces1, faces2]) if len(faces1) and len(faces2) else \
                np.vstack([faces1, faces3]) if len(faces1) and len(faces3) else \
                np.vstack([faces2, faces3]) if len(faces2) and len(faces3) else \
                faces1 if len(faces1) else faces2 if len(faces2) else faces3
    
    if len(all_faces) == 0:
        return 0, img
    
    # הסרת כפילויות
    final_faces = []
    for (x1, y1, w1, h1) in all_faces:
        is_duplicate = False
        for (x2, y2, w2, h2) in final_faces:
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            area1 = w1 * h1
            area2 = w2 * h2
            smaller_area = min(area1, area2)
            
            if overlap_area > 0.5 * smaller_area:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_faces.append([x1, y1, w1, h1])
    
    confirmed_faces = []
    for (x, y, w, h) in final_faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        aspect_ratio = float(w) / h
        relative_size = (w * h) / (img.shape[0] * img.shape[1])
        
        if (0.4 < aspect_ratio < 1.6 and
            0.005 < relative_size < 0.3 and
            (len(eyes) >= 1 or relative_size > 0.02)):
            confirmed_faces.append([x, y, w, h])
    
    # Blur faces
    for face in confirmed_faces:
        img = blur_face(img, face)
    
    return len(confirmed_faces), img

if __name__ == "__main__":
    # Example usage
    image_path = "my_img.jpeg"
    img = cv2.imread(image_path)
    num_faces, processed_img = detect_faces(img)
    print(f"מספר הפנים שזוהו וטושטשו: {num_faces}")
    cv2.imwrite("blurred_" + image_path, processed_img)
