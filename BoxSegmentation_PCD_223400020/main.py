import cv2
print(cv2.__version__)
print(hasattr(cv2, "TrackerCSRT_create"))

# Harus Install Package opencv-contrib-python
def create_tracker():
    return cv2.TrackerCSRT_create()


cap = cv2.VideoCapture(0)
tracker = None
tracking = False
background_frame = None

print("Mulai otomatis... Tunggu objek muncul untuk tracking.")
print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if background_frame is None:
        background_frame = frame_blur.copy()
        continue

    if not tracking:
        # Deteksi perubahan (perbedaan antara frame sekarang dan background)
        diff = cv2.absdiff(background_frame, frame_blur)
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1500:  # Hindari noise kecil
                x, y, w, h = cv2.boundingRect(largest)
                bbox = (x, y, w, h)
                tracker = create_tracker()
                tracker.init(frame, bbox)
                tracking = True
                print("Objek terdeteksi, tracking dimulai.")
    else:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking aktif", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking gagal", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            tracking = False
            tracker = None
            background_frame = frame_blur.copy()
            print("Tracking gagal. Menunggu objek baru...")

    cv2.imshow("Auto Tracking", frame)

cap.release()
cv2.destroyAllWindows()
