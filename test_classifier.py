import cv2
import mediapipe as mp
import pickle
import numpy as np

cam = cv2.VideoCapture(0)
labels_dict = {0: '0', 1: '1', 2: '2'}


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

    # Array per la posizione instantanea del landmark
    x_ = []
    y_ = []
    hight, width, _ = frame.shape


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # trova tutti i landmarks dell'immagine
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Considera solo la prima mano
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
        x_tmp = []
        y_tmp = []
            
        # Raccoglie X,Y della mano
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_tmp.append(x)
            y_tmp.append(y)

        # Normalizza sottraendo il valore minimo
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_tmp)) # Distanza relativa dal bordo sinistro della mano
            data_aux.append(y - min(y_tmp)) # Distanza relativa dal bordo superiore della mano
            x_.append(x)
            y_.append(y)
        
        # Cordinate della mano
        x1 = int(min(x_) * width)
        y1 = int(min(y_) * hight)
        x2 = int(max(x_) * width)
        y2 = int(max(y_) * hight)

        # prediction è una lista di un solo elemento
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Disegna prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows() 