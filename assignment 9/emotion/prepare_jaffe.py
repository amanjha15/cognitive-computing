"""Sort the JAFFE dataset into one folder per emotion and crop each face.

Usage: unzip jaffe.zip, put this file next to the unzipped folder, then run
    python prepare_jaffe.py
Output: jaffe_sorted/<emotion>/<name>.png  (ready to upload to LandingLens)
"""
import os
import cv2

SOURCE_DIR = "jaffe"           # folder with the .tiff files (searched recursively)
OUTPUT_DIR = "jaffe_sorted"
FACE_SIZE = 128                # must match FACE_SIZE in the notebook
MARGIN = 0.10                  # must match the margin in the notebook's prepare_face

CODES = {"AN": "angry", "DI": "disgust", "FE": "fear", "HA": "happy",
         "NE": "neutral", "SA": "sad", "SU": "surprise"}

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
counts, no_face = {}, 0

for root, _, files in os.walk(SOURCE_DIR):
    for fname in files:
        if not fname.lower().endswith((".tiff", ".tif")):
            continue
        parts = fname.split(".")
        emotion = CODES.get(parts[1][:2].upper()) if len(parts) > 2 else None
        if emotion is None:
            continue

        gray = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            mx, my = int(w * MARGIN), int(h * MARGIN)
            gray = gray[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]
        else:
            no_face += 1               # keep the full image if no face is found

        out_dir = os.path.join(OUTPUT_DIR, emotion)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(fname)[0] + ".png"),
                    cv2.resize(gray, (FACE_SIZE, FACE_SIZE)))
        counts[emotion] = counts.get(emotion, 0) + 1

print("Images per emotion:", dict(sorted(counts.items())))
print("Total:", sum(counts.values()), "| no face detected (kept full image):", no_face)
