import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./haberman.csv')
X = np.array(dataset.iloc[:, :-1])
y = np.array(dataset.iloc[:, -1])




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)



# Predict
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))


# Compute Confusion Metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

print(acc)