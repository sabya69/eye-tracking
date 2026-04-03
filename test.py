import cv2

def count_cameras(max_tested=10):
    available_cameras = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows backend
        
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    return available_cameras


cams = count_cameras()

print("Connected Cameras Indexes:", cams)
print("Total Cameras Found:", len(cams))