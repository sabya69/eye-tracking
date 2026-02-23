#  Real-Time Attention Tracking System

A real-time Attention Monitoring System built using MediaPipe and OpenCV that tracks eye movement, blink rate, head pose, and calculates an overall attention score.

This project demonstrates advanced Computer Vision techniques including facial landmark detection, iris tracking, blink detection (EAR), head pose estimation, and real-time analytics dashboard generation.

---

##  Overview

The system captures live webcam input and analyzes:

-  Eye gaze direction
-  Blink frequency
-  Head orientation (Yaw & Pitch)
-  Rolling attention score
-  Drowsiness detection

It logs session data and generates a visual performance report at the end.

This project can be used for:

- Student attention monitoring
- Online exam supervision
- Driver drowsiness detection
- Human-computer interaction research
- Productivity tracking systems

---

##  Key Features

- ✅ Real-time face landmark detection (468 landmarks)
- ✅ Iris localization using MediaPipe Face Landmarker
- ✅ Blink detection using Eye Aspect Ratio (EAR)
- ✅ Head pose estimation (Yaw & Pitch)
- ✅ Live gaze direction detection (LEFT / RIGHT / CENTER)
- ✅ Rolling attention score (0–100%)
- ✅ Drowsiness alert system
- ✅ CSV session logging
- ✅ Automatic dashboard generation (graphs & analytics)
- ✅ Clean OpenCV display interface
- ✅ TensorFlow log suppression

---

##  Technologies Used

- Python
- OpenCV
- MediaPipe Tasks API
- NumPy
- Pandas
- Matplotlib
- Seaborn
- CSV
- Winsound (for alert sound)

---

##  How It Works

### 1️⃣ Face & Landmark Detection
MediaPipe Face Landmarker detects 468 facial landmarks from live webcam frames.

### 2️⃣ Iris Detection
Iris landmarks are extracted and a minimum enclosing circle is used to calculate iris center.

### 3️⃣ Blink Detection (EAR Method)
Eye Aspect Ratio (EAR) is calculated using vertical and horizontal eye landmarks.

If:
- EAR < threshold → blink detected
- Eye closed > 2 seconds → Drowsiness alert triggered

### 4️⃣ Gaze Detection
The iris center is compared with eye corners to determine:

- LEFT
- RIGHT
- CENTER

### 5️⃣ Head Pose Estimation
Facial transformation matrix is used to compute:

- Yaw (left/right rotation)
- Pitch (up/down tilt)

### 6️⃣ Attention Score Calculation

The attention score is calculated based on:

- Blink frequency
- Head orientation stability
- Gaze consistency
- Gaze variance

Score Range:

-  HIGH: ≥ 70%
-  MEDIUM: 40–69%
-  LOW: < 40%

---

##  Output

During runtime:
- Live webcam feed with overlays
- Gaze direction display
- Blink count
- Head pose values
- Attention percentage

After session ends:
- CSV log file
- Attention score graph
- Blink timeline
- Gaze distribution chart
- Session summary statistics

---

##  Project Structure

```
attention-tracker/
│
├── attention_tracker.py
├── face_landmarker.task
├── gaze_log.csv
├── session_report.png
├── requirements.txt
└── README.md
```

---

##  Installation & Running

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Project

```
python attention_tracker.py
```

Press **ESC** to exit the application.

---

##  Example Use Cases

-  Online classroom attention monitoring
-  Driver fatigue detection
-  Productivity analysis
-  Behavioral research
-  Human-computer interaction

---

##  Future Improvements

- Machine learning-based gaze classification
- Multi-person attention tracking
- Web-based dashboard integration
- Cloud-based session analytics
- Emotion detection integration
- Cursor control using eye movement

---

##  Learning Outcomes

- Real-time computer vision processing
- Facial landmark detection
- Geometric feature extraction
- Attention modeling logic
- Data logging & visualization
- Human behavioral analysis using computer vision

---

## 👨 Author

Sabyasachi Das Biswas  
Computer Vision Mini Project  
2026

---

