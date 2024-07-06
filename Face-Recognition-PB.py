import face_recognition
import os
import cv2
import numpy as np
import math
import sys 
import threading

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100 , 2)) +'%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) *100
        return str(round(value,2)) + '%'
    
class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        directory = "Face_Recognition_PB\Images"
        for image_file in os.listdir(directory):
            image_path = os.path.join(directory, image_file)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(os.path.splitext(image_file)[0])

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names_and_confidences = []
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                face_names_and_confidences.append(((top, right, bottom, left), name, confidence))

        return face_names_and_confidences


def frame_capture(video_capture, frame_buffer, lock):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        with lock:
            frame_buffer[0] = frame.copy()

def main():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        sys.exit("Video source not found")

    face_recognizer = FaceRecognition()
    frame_buffer = [None]
    frame_lock = threading.Lock()

    # Start the frame capture thread
    frame_thread = threading.Thread(target=frame_capture, args=(video_capture, frame_buffer, frame_lock))
    frame_thread.daemon = True
    frame_thread.start()

    while True:
        with frame_lock:
            frame = frame_buffer[0]
           

        if frame is not None:
            # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            face_names_and_confidences = face_recognizer.recognize_faces(frame)

            for (top, right, bottom, left), name, confidence in face_names_and_confidences:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            
                cv2.putText(frame, f"{name} ({confidence})", (left + 6, bottom - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1)

            
            cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

