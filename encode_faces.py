import os
import cv2
import face_recognition
import pickle

# Path to the dataset
dataset_path = r"C:\Users\ishan\OneDrive\Desktop\face recog\workfromstart\New folder"
# Path to save the encodings
encodings_path = 'encodings.pickle'

known_encodings = []
known_names = []

# Loop over each person in the dataset
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    # Loop over each image for the current person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb, model='hog')

        # Compute facial encodings
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the encodings and names to disk
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encodings have been saved to 'encodings.pickle'")
