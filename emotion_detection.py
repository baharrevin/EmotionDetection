import cv2
import pygame
from deepface import DeepFace

cap = cv2.VideoCapture(1)

pygame.mixer.init()
# pygame: Ses dosyalarını çalmak için pygame.mixer modülü.


duygu_cevirisi = {
    "angry": "Kizgin",
    "disgust": "Tiksinti",
    "fear": "Korku",
    "happy": "Mutlu",
    "sad": "Uzgun",
    "surprise": "Saskin",
    "neutral": "Notr"
}
# ingilizce bir şekilde geldiği için türkçelerini verdik

duygu_sesleri = {
    "happy": "Görüntü İşleme 2_happy.mp3",
    "sad": "Görüntü İşleme 2_sad.mp3"
}


son_duygu = None
# Aynı duygunun sesi tekrar tekrar çalınmasın diye önceki duyguyu hatırlamak için


while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # enforce_detection=False: Yüz algılanamazsa hata vermesin

        for face in analysis:
            x,y,w,h = face['region']['x'],face['region']['y'],face['region']['w'],face['region']['h']
        
        emotion = face['dominant_emotion']
        turkce_emotion = duygu_cevirisi.get(emotion,emotion)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,turkce_emotion,(x,y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
        # Yüzün etrafına yeşil dikdörtgen çizilir.
        # Üstüne duygu yazısı eklenir (örneğin "Mutlu").

        if emotion == "happy":
            cv2.putText(frame, "You look happy, come on, join the music :)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2, cv2.LINE_AA)
        if emotion == "sad":
            cv2.putText(frame, "You look sad, come on, sing along to the music and find happiness, everything will be fine, smile :)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2, cv2.LINE_AA)
        if emotion == "surprise": 
            cv2.putText(frame, "You look surprised", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2, cv2.LINE_AA)
        if emotion == "fear":
            cv2.putText(frame, "You look scared", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2, cv2.LINE_AA)
            
        if emotion in duygu_sesleri and emotion != son_duygu:
            pygame.mixer.music.load(duygu_sesleri[emotion])
            pygame.mixer.music.play()
            son_duygu = emotion

    except Exception as e:
        print("Hata: ",e)
    
    cv2.imshow("Gercek Zamanli Yuz Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()