import numpy as np
import pandas as pd
import tensorflow as tf

"""## Part 1 - Data Preprocessing

### Importing the dataset
"""

dataset = pd.read_csv('./Social_Network_Ads.csv')
X = np.array(dataset.iloc[:, :-1])
y = np.array(dataset.iloc[:, -1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# remember to fit the scale itself to only the train set,
# and apply it to test set, to prevent info leakage
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Part 2 - Building the ANN

### Initializing the ANN
"""

# create an instance of the sequential ann class
ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

# add Dense layer as a first hidden layer to the ann

# units are number of neurons in the first hidden layer we want
# we just guess and experiment with this value, no good rule of thumb
units = 6
# set rectifier as the activation function for each neuron, 
# in ANN we must use rectifier activation function
activationFunction = 'relu'
# create the first layer
firstLayer = tf.keras.layers.Dense(units=units, activation=activationFunction)
#add it
ann.add(firstLayer)

"""### Adding the second hidden layer"""

# add second layer which here is the same as first
ann.add(firstLayer)


"""### Adding the output layer"""

# since our output is binary we only need on neuron
units = 1
# we need sigmoid activation function so we get probabilistic prediction
activation_function = 'sigmoid' # use softmax for non binary clasification
outputLayer = tf.keras.layers.Dense(units=units, activation=activation_function)
ann.add(outputLayer)

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

optimizer = 'adam'  # update the weights through stacastic gradient discent to reduce the loss  
loss_function = 'binary_crossentropy' # always used for binary clasification use categorical_crossentropy for non binary clasification
metrics = ['accuracy']
ann.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

"""### Training the ANN on the Training set"""

# this will split the training set into batches, training model through batches at a time instead of one by one
batch_size = 8
# how many times the ann will go over the whole test set
epochs = 100

ann.fit(x=X_train, y=y_train, batch_size=batch_size, epochs= epochs)

## Part 4 - Making the predictions and evaluating the model


# Therefore, our ANN model predicts that this customer stays in the bank!

# convert predictions to binary
y_pred =  ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Compute Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Compute Accuracy Score
acc = accuracy_score(y_test, y_pred)

print(acc)
