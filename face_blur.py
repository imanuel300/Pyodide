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
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    all_faces = []
    angles = [-15, 0, 15]  # פחות זוויות לבדיקה
    
    height, width = gray.shape
    center = (width // 2, height // 2)
    
    # פרמטרים מותאמים לביצועים טובים יותר
    scale_factors = [1.1]  # רק ערך אחד
    min_neighbors_options = [4]  # רק ערך אחד
    
    for angle in angles:
        if angle != 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_gray = cv2.warpAffine(gray, M, (width, height))
        else:
            rotated_gray = gray
        
        # זיהוי עם שני המסווגים העיקריים
        for cascade in [face_cascade, face_cascade_alt]:
            faces = cascade.detectMultiScale(
                rotated_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                maxSize=(width//2, height//2),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces):
                if angle != 0:
                    for i, (x, y, w, h) in enumerate(faces):
                        corners = np.array([
                            [x, y],
                            [x + w, y],
                            [x, y + h],
                            [x + w, y + h]
                        ])
                        
                        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        rotated_corners = []
                        for corner in corners:
                            point = np.array([corner[0], corner[1], 1])
                            rotated_point = np.dot(M_inv, point)
                            rotated_corners.append(rotated_point[:2])
                        rotated_corners = np.array(rotated_corners)
                        
                        x = int(np.min(rotated_corners[:, 0]))
                        y = int(np.min(rotated_corners[:, 1]))
                        w = int(np.max(rotated_corners[:, 0]) - x)
                        h = int(np.max(rotated_corners[:, 1]) - y)
                        faces[i] = [x, y, w, h]
                
                all_faces.extend(faces)
    
    if len(all_faces) == 0:
        return 0, img
    
    all_faces = np.array(all_faces)
    
    # הסרת כפילויות עם סף חפיפה מותאם
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
            
            if overlap_area > 0.4 * smaller_area:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_faces.append([x1, y1, w1, h1])
    
    # אימות פנים מהיר יותר
    confirmed_faces = []
    for (x, y, w, h) in final_faces:
        aspect_ratio = float(w) / h
        relative_size = (w * h) / (img.shape[0] * img.shape[1])
        
        # בדיקת עיניים רק אם היחסים סבירים
        if (0.35 < aspect_ratio < 1.65 and 0.003 < relative_size < 0.35):
            if relative_size > 0.02:  # אם הפנים גדולות מספיק, לא צריך לבדוק עיניים
                confirmed_faces.append([x, y, w, h])
            else:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 1:
                    confirmed_faces.append([x, y, w, h])
    
    # טשטוש הפנים
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
