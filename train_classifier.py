import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Preparazione data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divide in training set e test set 
# 20% del data è test set, mischia i dati, mantiene le proporzioni tra le labels di partenza
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


# Iizializza il modello, fa il training e poi fa la prediction
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Check della performance del modello
score = accuracy_score(y_predict, y_test)

print('{}% sono stati classificati correttamente'.format(score *  100))

# Salviamo il modello con un dictionary
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()