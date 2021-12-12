import os

import cv2
import face_recognition
import numpy as np
import pandas as pd

student_images: list[str] = []
classes: list[str] = []
path: str = '../dataset'
students: list[str] = os.listdir(path)
student_details: dict[str, dict[str, str]] = {}

for student in students:
    current_image = cv2.imread(f"{path}/{student}")
    student_images.append(current_image)
    classes.append(os.path.splitext(student)[0])


def find_encodings(images: list[str]) -> list[str]:
    encode_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
    return encode_list


encode_list_known = find_encodings(student_images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, .25, .25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_current_frame = face_recognition.face_locations(img_small)
    encodesCurFrame = face_recognition.face_encodings(img_small, face_current_frame)

    for encode_face, face_loc in zip(encodesCurFrame, face_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            details: list[str] = classes[match_index].split(",")
            name: str = details[0].strip()
            college_id: str = details[1].strip()
            year: str = details[2].strip()
            batch: str = details[3].strip()
            student_details[college_id] = {
                "college id": college_id,
                "name": name,
                "year": year,
                "batch": batch
            }
            name = name.upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow("Attendance", img)
    cv2.waitKey(1)

    # Exporting to csv
    df = pd.DataFrame.from_dict(student_details, orient="index")
    df.to_csv("att.csv")

    # Press exit button to close program
    if cv2.getWindowProperty("Attendance", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
