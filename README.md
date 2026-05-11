Hand Gesture Recognition System
Questo progetto implementa una pipeline completa per il riconoscimento dei gesti delle mani in tempo reale. Il sistema utilizza la visione artificiale per l'estrazione delle feature e il machine learning per la classificazione.

- Architettura del Sistema
Il progetto è diviso in tre fasi principali:

Data Collection: Acquisizione di campioni d'immagine tramite webcam.

Feature Extraction: Estrazione dei landmark geometrici della mano tramite MediaPipe.

Classification: Addestramento di un modello Random Forest per il riconoscimento dei gesti.

- Requisiti Tecnici

Il progetto è ottimizzato per Python 3.11. Per garantire la compatibilità con MediaPipe su questa versione, è necessario utilizzare le versioni specifiche indicate in dependency.txt.

- Struttura del Software

1. Acquisizione Dati (collect_data.py)
Lo script gestisce l'input video dalla webcam (cv2.VideoCapture).
Genera una struttura di directory in ./data/ per ogni classe.

Implementa un sistema di "Ready-Check": l'utente preme 'Q' per avviare la cattura una volta posizionata la mano.

Cattura 50 frame per classe per garantire un dataset bilanciato.

2. Preprocessing e Landmark Extraction (create_dataset.py)
In questa fase, le immagini grezze vengono trasformate in vettori numerici.

MediaPipe Hands: Utilizzato in static_image_mode=True. L'algoritmo rileva 21 landmark 3D (punti di interesse) per ogni mano.

Vettorizzazione: Per ogni immagine, vengono estratti i valori (x,y) di ogni landmark, creando un feature vector di 42 elementi.

Storage: I dati processati e le relative etichette vengono serializzati in data.pickle.

3. Training del Modello (train_classifier.py)
Il cuore predittivo del sistema.

Model: Utilizza un RandomForestClassifier della suite Scikit-Learn.

Data Splitting: Il dataset viene diviso (80% training, 20% test) con stratificazione (stratify=labels) per assicurare che il modello veda una distribuzione equa di ogni gesto durante l'apprendimento.

Validazione: Calcola l'accuratezza confrontando le predizioni con i valori reali del test set.

Export: Il modello finale viene salvato come model.p.

- Come Eseguire il Progetto
Raccogli le immagini:
  Bash
  python collect_data.py

Estrai i landmark:
  Bash
  python create_dataset.py

Addestra l'intelligenza artificiale:
  Bash
  python train_classifier.py

- Note sulla Manutenzione e Debug
Dimensione dei dati: Assicurati che durante la fase di creazione del dataset venga rilevata una sola mano per immagine. Se vengono rilevate più mani, la lunghezza del vettore data_aux varierà, causando errori nel classificatore Random Forest.

Ambiente Virtuale: Si raccomanda vivamente l'uso di un venv per evitare conflitti tra versioni diverse di OpenCV e MediaPipe.

Git: Il file data.pickle e la cartella ./data/ sono stati aggiunti al .gitignore per evitare l'upload di file binari pesanti e dataset locali.

Autore: Andrea Parlati
