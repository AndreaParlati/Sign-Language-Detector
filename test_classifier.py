import cv2
import mediapipe as mp
import pickle
import numpy as np

cam = cv2.VideoCapture(0)

# Carica il modello del riconoscimento
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


# Carica il modello per il tracciamento della mano
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inizializza l'algoritmo e indica:
#  - nel primo parametro che stiamo analizzando delle immagini in flusso continuo
#  - nel secondo parametro indica il livello di sicurezza prima di procedere con l'analisi (se almeno al 30% c'è una mano allora analizzerà)
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

while True:
    data_aux = []
    ret, frame = cam.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # trova tutti i landmarks dell'immagine
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Disegna i landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
                )
        # Salva le posizioni dei landmarks in un array ausiliario
        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
   
        prediction = model.predict([np.asarray(data_aux)])

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows() 