import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import multiprocessing
import os 

def face_extraction(img):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    h,w,_ = img.shape 
    face_detection_results = face_detection.process(img[:,:,::-1])
    if face_detection_results.detections:
        for face_no, face in enumerate(face_detection_results.detections):
            face_data = face.location_data
            faces = [[int(face_data.relative_bounding_box.xmin*w),int(face_data.relative_bounding_box.ymin*h),int(face_data.relative_bounding_box.width*w),int(face_data.relative_bounding_box.height*h)]]
            for x, y, w, h in faces:
                cropped_img = img[y-int(h/3):y + 13*int(h/12), x:x + w]
    return cropped_img

folder = r"pics"
os.makedirs("extracted_face1")
os.makedirs("Error_images1")
for image in os.listdir(folder):
    path = os.path.join(folder,image)
    try:
        img = cv2.imread(path)
        extracted_face = face_extraction(img)
        cv2.imwrite(f"extracted_face1/{image}.jpg",extracted_face)
    except Exception as e:
        img = cv2.imread(path)
        cv2.imwrite(f"Error_images1/{image}.jpg",img)