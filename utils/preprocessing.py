"""
preprocessing.py
Face detection, grayscale conversion, and resizing utilities using OpenCV.
"""
import cv2
import numpy as np
import os
from pathlib import Path

# Load Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

IMG_SIZE = (100, 100)


def detect_and_crop_face(image):
    """
    Detect a face in the image and return the cropped face region.
    If no face is detected, return the entire image resized.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y+h, x:x+w]
    else:
        face = gray

    face_resized = cv2.resize(face, IMG_SIZE)
    return face_resized


def process_uploaded_images(upload_dir):
    """
    Process all uploaded images organized in subfolders (one per person).
    Returns:
        images: list of numpy arrays (grayscale, resized)
        labels: list of string labels
        label_names: sorted unique label names
        preview_paths: list of relative paths for preview
    """
    images = []
    labels = []
    preview_paths = []

    if not os.path.exists(upload_dir):
        return images, labels, [], preview_paths

    label_names = sorted([
        d for d in os.listdir(upload_dir)
        if os.path.isdir(os.path.join(upload_dir, d))
    ])

    for label in label_names:
        person_dir = os.path.join(upload_dir, label)
        for fname in os.listdir(person_dir):
            fpath = os.path.join(person_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = fname.lower().split('.')[-1]
            if ext not in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'):
                continue

            img = cv2.imread(fpath)
            if img is None:
                continue

            face = detect_and_crop_face(img)
            images.append(face)
            labels.append(label)

            # Save processed face for preview
            processed_dir = os.path.join(upload_dir, '_processed', label)
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, fname)
            cv2.imwrite(processed_path, face)
            preview_paths.append(f'_processed/{label}/{fname}')

    return images, labels, label_names, preview_paths


def process_single_image(image_bytes):
    """
    Process a single uploaded test image (from bytes).
    Returns the preprocessed grayscale face array.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    face = detect_and_crop_face(img)
    return face
