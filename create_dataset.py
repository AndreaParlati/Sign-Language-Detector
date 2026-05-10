import os
import mediapipe as mp
import cv2
import pickle


# carica il modello per il tracciamento della mano
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mo_drawing_styles = mp.solutions.drawing_styles

# Inizializza l'algoritmo e indica:
#  - nel primo parametro che stiamo analizzando delle immagini statiche e non un flusso continuo
#  - nel secondo parametro indica il livello di sicurezza prima di procedere con l'analisi (se almeno al 30% c'è una mano allora analizzerà)
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
# scorre tutte le immagini di tutte le dir della directory data e le converte da BGR a RGB per poterle usare con mediapipe
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # trova tutti i landmarks dell'immagine
        results = hands.process(img_rgb)

        # crea un array di landmarks trovati
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            
            # Crea una lista di array di landmarks che rappresentano le immagini
            data.append(data_aux)
            labels.append(dir_)        

# Salva i dati creati
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()