import cv2
import cv2.data

# OpenCV’nin Haar Cascade tabanlı hazır yüz algılama modelini yüklüyoruz.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1)

# ret: Kameradan görüntü alınıp alınamadığını belirtir (True/False).
# frame: Kameradan alınan tek bir görüntü karesi (frame).

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü bulunamadı!")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Gri tonlama, işlem süresini hızlandırır ve algoritmanın çalışmasını kolaylaştırır.

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
    # detectMultiScale() fonksiyonu gri tonlamalı görüntüde yüzleri bulur.
    # scaleFactor=1.3: Algoritmanın görüntüyü kaç ölçek küçülteceği (detayda yüz aramak için).
    # minNeighbors=5: Bir bölgenin yüz olarak kabul edilmesi için kaç komşu algılanması gerektiği (yanlış pozitifleri azaltır).
    # minSize=(30,30): Algılanacak yüzlerin minimum boyutu (piksel cinsinden).

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Real Time Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()