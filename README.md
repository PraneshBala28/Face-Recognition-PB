# Face Recognition using OpenCV and face_recognition

This project implements real-time face recognition using a webcam feed. It leverages the `opencv-python` and `face_recognition` libraries to detect and recognize faces in live video streams. The system captures frames from the video feed, identifies faces using the `face_recognition` library's facial recognition capabilities, and compares them against a database of known faces stored in the "Images" directory. Each recognized face is annotated with its name and confidence level based on facial recognition results.

The project incorporates multi-threading to enhance performance, with one thread dedicated to capturing frames and another to process and recognize faces concurrently. This approach ensures efficient handling of real-time video input, making it suitable for applications requiring immediate response to detected faces.

### Key Features

- **Real-Time Face Detection and Recognition:** Uses `opencv-python` for real-time video capture and `face_recognition` for accurate face recognition.
  
- **Face Database Management:** Loads and encodes known faces from images stored in the "Images" directory, enabling recognition against a pre-defined set of individuals.
  
- **Confidence Level Calculation:** Calculates and displays confidence levels for recognized faces, aiding in decision-making based on the reliability of facial recognition results.

This project is suitable for applications in security systems, attendance tracking, and personalized user experiences where real-time face recognition is required. It provides a robust foundation for further customization and integration into larger systems or applications.


## Requirements

- Python 3.x
- OpenCV (`opencv-python`) version 4.10.0.84
- face_recognition version 1.3.0

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/PraneshBala28/Face-Recognition-PB.git
    cd face-recognition-pb
    ```

2. Install the required dependencies:
    ```bash
    pip install opencv-python==4.10.0.84
    pip install face-recognition==1.3.0
    ```

## Usage

1. Run the face recognition script:
    ```bash
    python face_recognition.py
    ```

2. Press `q` to quit the program.
---

*Developed by Pranesh Balaji.*
