import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# face_detection: Mediapipe’in yüz algılama modülüdür.
# drawing_utils: Tespit edilen yüzün üzerine grafik (kutu, anahtar noktalar) çizebilmemizi sağlar.

cap = cv2.VideoCapture(1)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
# model_selection=0: Yakın mesafeli yüzler için optimize edilmiş model
# (1 dersen daha uzaktaki yüzleri de algılayabilir).
# min_detection_confidence=0.5: %50 güvenin altındaki tahminleri görmezden gelir.
    
    while cap.isOpened(): # Kamera açık olduğu sürece döngü devam eder.
        ret, frame = cap.read()
        # ret: Kameradan görüntü alınıp alınamadığını belirtir (True/False).
        # frame: Kameradan alınan tek bir görüntü karesi (frame).

        if not ret:
            print("Kamera görüntüsü alınamadı!")
            break

        frame = cv2.flip(frame,1)
        # kameradan gelen görüntüyü sağ-sol ters çevirir. ayna görüntüsü
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # OpenCV BGR formatında çalışır, Mediapipe ise RGB ister. dönüşüm yapıyoruz.

        result = face_detection.process(rgb_frame)
        # Mediapipe modeline görüntüyü veriyoruz.

        if result.detections:
            for detection in result.detections:
                mp_drawing.draw_detection(frame,detection)
        # draw_detection() ile yüzün etrafına kutu ve işaretler çizilir.

        cv2.imshow("Gercek Zamanli Yuz Algilama", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()