import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import tensorflow as tf

# Get the current working directory
current_directory = os.getcwd()

# Go back one level to the parent directory
parent_directory = os.path.dirname(current_directory)

# Open a different folder
desired_folder = "processed_data"  
folder_path = os.path.join(parent_directory, desired_folder)

# Check if the folder exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    file_to_read = "filtered_dataset_v2.csv" 
    file_path = os.path.join(folder_path, file_to_read)
    
    # Check if the file exists
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"File '{file_to_read}' does not exist in folder '{desired_folder}'.")
else:
    print(f"Folder '{desired_folder}' does not exist in the parent directory.")

feature_cols = ["Temperature (C)", "Humidity", "Pressure (millibars)"]
encoder = LabelEncoder()
X = df[feature_cols]
Y = encoder.fit_transform(df.Summary)

# split X and y into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regresison
lr = LogisticRegression(random_state=16,solver='lbfgs',max_iter=1000).fit(X_train_scaled, Y_train)

Y_pred = lr.predict(X_test_scaled)
cnf_matrix_lr = metrics.confusion_matrix(Y_test, Y_pred)
lr.score(X_test_scaled,Y_test)
print(classification_report(Y_test, Y_pred))

# One vs One SVC Approach

svc = svm.SVC(decision_function_shape="ovo")
svc.fit(X_train_scaled, Y_train)

Y_pred_svc = svc.predict(X_test_scaled)
cnf_matrix_svc = metrics.confusion_matrix(Y_test, Y_pred_svc)
svc.score(X_test_scaled,Y_test)
print(classification_report(Y_test, Y_pred_svc))

# One Vs Rest Approach

lin_svc = svm.LinearSVC(dual="int").fit(X_train_scaled,Y_train)
Y_pred_lin_svc = svc.predict(X_test_scaled)

cnf_matrix_lin_sv = metrics.confusion_matrix(Y_test, Y_pred_lin_svc)
lin_svc.score(X_test_scaled, Y_test)
print(classification_report(Y_test, Y_pred_lin_svc))

# KNN
knn = KNeighborsClassifier(n_neighbors=4).fit(X_train_scaled,Y_train)
Y_pred_knn = knn.predict(X_test_scaled)

cnf_matrix_knn = metrics.confusion_matrix(Y_test, Y_pred_knn)
knn.score(X_test_scaled, Y_test)
print(classification_report(Y_test, Y_pred_knn))

# Gaussian Naive Bayes
gnb = GaussianNB().fit(X_train_scaled, Y_train)
Y_pred_gnb = gnb.predict(X_test_scaled)

cnf_matrix_gnb = metrics.confusion_matrix(Y_test, Y_pred_gnb)
gnb.score(X_test_scaled, Y_test)
print(classification_report(Y_test, Y_pred_gnb))

# Decision Tree
dt = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, Y_train)
Y_pred_dt = dt.predict(X_test_scaled)

cnf_matrix_dt = metrics.confusion_matrix(Y_test, Y_pred_dt)
dt.score(X_test_scaled, Y_test)
print(classification_report(Y_test, Y_pred_dt))

# Random Forest
rf = RandomForestClassifier(random_state=0).fit(X_train_scaled, Y_train)
Y_pred_rf = rf.predict(X_test_scaled)

cnf_matrix_rf = metrics.confusion_matrix(Y_test, Y_pred_rf)
rf.score(X_test_scaled, Y_test)
print(classification_report(Y_test, Y_pred_rf))

# Deep Learning Methods

# Hyperparameters for ANN 
num_classes = 4
epochs = 150
input_dimension = X_train.shape[1]
batch_size = 64
learning_rate = 0.001

y_train_encoded = tf.keras.utils.to_categorical(Y_train, num_classes)
y_test_encoded = tf.keras.utils.to_categorical(Y_test, num_classes)

x_train_reshaped = np.expand_dims(X_train_scaled, axis=2)
x_test_reshaped = np.expand_dims(X_test_scaled, axis=2)

# Callback function to avoid overfitting
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.90) and (logs.get('accuracy')>0.95):
            print("\nValidation and training accuracies are high so cancelling training!")
            self.model.stop_training = True

# Feed Forward Neural Network
# Architecture 1: 64-64-128-3 Feed Forward Neural Network
# Defining the ANN architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=input_dimension)) 
model.add(tf.keras.layers.Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dense(128, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) 
# Model Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Stopping early to avoid overfitting
stop_callback = myCallback()

# Training
history = model.fit(x_train_reshaped, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(x_test_reshaped, y_test_encoded), callbacks=[stop_callback])


# Architecture 2: 32-256-3 Feed Forward Neural Network with 'relu' and 'softmax'
# Defining the ANN architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=input_dimension)) 
model.add(tf.keras.layers.Dense(256, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) 
# Model Compilation
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Stopping early to avoid overfitting
stop_callback = myCallback()
# Training
history = model.fit(x_train_reshaped, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(x_test_reshaped, y_test_encoded), callbacks=[stop_callback])

test_loss_1, test_accuracy_1 = model.evaluate(x_test_reshaped, y_test_encoded, verbose=0)
print('Test Loss:', test_loss_1)
print('Test Accuracy:', test_accuracy_1)

# Using RNN
# Architecture 1: 32-64-128-3 RNN with 'relu' and 'softmax'
# Defining the RNN architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(32, activation='relu', input_shape=(input_dimension, 1)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Model Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Stopping early to avoid overfitting
stop_callback = myCallback()
# Training
history = model.fit(x_train_reshaped, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(x_test_reshaped, y_test_encoded), callbacks=[stop_callback])
