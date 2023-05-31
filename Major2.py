#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import librosa
# import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Conv2D, LSTM, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dense, Dropout, Activation, SimpleRNN, TimeDistributed, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import TruncatedNormal
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

import IPython.display as ipd


# In[2]:


folder_path = r'D:\Anamika Files\College\SEM8\Major 2\Sugandha Dataset\Reduced Features\Data'
 
# Define the function to extract MFCC features from audio files
def extract_mfcc_features(file_path):
    try:
        if file_path.endswith('.wav'):
            audio, sr = librosa.core.load(file_path, sr=22050, mono=True)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T,axis=0)
            return mfccs_processed
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

# Load the audio files and extract MFCC features
audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  # List of paths to audio files
mfcc_features = []
for file_path in audio_files:
    mfccs = extract_mfcc_features(file_path)
    if mfccs is not None:
        mfcc_features.append(mfccs)
mfcc_features = np.asarray(mfcc_features)


# In[3]:


path = "D:/Anamika Files/College/SEM8/Major 2/Sugandha Dataset/Reduced Features/Data"
df = pd.read_csv(path + "/reducedfeatures.csv")
df


# In[4]:


y = df['Phase']
x = df.drop('Phase',axis=1)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(mfcc_features, y, test_size=0.2, random_state=42)


# In[6]:


# Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


# Train the SVM model with regularization
C = 1.0  # Regularization parameter
svc = SVC(kernel='linear', C=C)
svc.fit(X_train, y_train)


# In[8]:


# Make predictions on the testing set
y_pred = svc.predict(X_test)


# In[9]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[10]:


#Random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[11]:


y_pred = rf.predict(X_test)


# In[12]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[13]:


#k neighbors classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)


# In[14]:


y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# In[15]:


#naive bayes classifier
clfnb = GaussianNB()
clfnb.fit(X_train,y_train)


# In[16]:


y_pred = clfnb.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[44]:


# Define function to extract mel spectrogram
def extract_melspectrogram(audio_file_path):
    # Load audio file using librosa
    signal, sr = librosa.load(audio_file_path, sr=None)
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    # Convert to decibel scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize between -1 and 1
    norm_log_mel_spec = (log_mel_spec + 80) / 80
    return norm_log_mel_spec


# Set path to audio files folder
audio_files_path = folder_path

# Define list to store mel spectrograms 
mel_specs = []

# Loop through audio files folder
for filename in os.listdir(audio_files_path):
    if filename.endswith('.wav'):
        # Extract mel spectrogram
        mel_spec = extract_melspectrogram(os.path.join(audio_files_path, filename))
        # Append mel spectrogram to list
        mel_specs.append(mel_spec)
    
# Create list of labels from excel file
label_list = y.tolist()

max_len = max(len(spec) for spec in mel_specs)

# Pad or truncate all spectrograms to the same length
fixed_length_spectrograms = [
    librosa.util.fix_length(spec, size=max_len)
    for spec in mel_specs
]

# Convert lists to numpy arrays
mel_specs = np.array(fixed_length_spectrograms)
y = np.array(label_list)


# In[18]:


print(mel_specs.shape)
print(y.shape)


# In[45]:


# Split data into training and testing sets
x_train, x_test, Y_train, Y_test = train_test_split(mel_specs, y, test_size=0.2, random_state=42)


# In[20]:


print(mel_specs.shape)
print(y.shape)


# In[21]:


#Implementing CNN

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(mel_specs.reshape(235,128,128,1), y, epochs=50, batch_size=32, validation_split=0.2)


# In[22]:


# Evaluate the model
loss, accuracy = model.evaluate(x_test.reshape(x_test.shape[0], 128, 128, 1), Y_test, verbose=0)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)


# In[23]:


print(mel_specs.shape)  # should return (num_samples, 128, 1)
print(y.shape)  # should return (num_samples, num_classes)


# In[24]:


# RNN

# Calculate number of samples to pad
#num_samples = mel_specs.shape[0]
#num_pad_samples = 128 - (num_samples % 128)

# Pad the input data with zeros
#pad_width = [(0, num_pad_samples), (0, 0), (0, 0)] if num_pad_samples > 0 else [(0, 0), (0, 0), (0, 0)]
#mel_specs = np.pad(mel_specs, pad_width, mode='constant')







# In[25]:


# Reshape input data
#mel_specs = np.reshape(mel_specs, (-1, 128, 1))
# Reshape the input data to be 1-dimensional
#mel_specs = tf.reshape(mel_specs, (mel_specs.shape[0], 128, -1))
#mel_specs = np.expand_dims(mel_specs, axis=-1)


# In[26]:


# Build the model
#model = Sequential()
#model = Sequential()
#model.add(SimpleRNN(units=128, input_shape=(128, 1)))
#model.add(Dense(units=10, activation='softmax'))


# In[27]:


# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#model.fit(mel_specs, y, epochs=50, batch_size=32, validation_split=0.2)


# In[46]:


print(mel_specs.shape)  # should return (num_samples, 128, 1)
print(y.shape)  # should return (num_samples, num_classes)
Y_train.shape
x_train.shape


# In[47]:


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train.shape


# In[48]:


# Evaluate the model
#loss, accuracy = model.evaluate(x_test.reshape(x_test.shape[0], 128, 128, 1), Y_test, verbose=0)
#print('Test loss:', loss)
#print('Test accuracy:', accuracy)
num_classes = 2 
# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)
Y_train.shape


# In[31]:


def ResNet34(x_train, Y_train, x_test, Y_test, num_classes, batch_size, epochs):
    def residual_block(inputs, filters, strides=1):
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        if strides == 1:
            shortcut = inputs
        else:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(inputs)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    input_shape = (128, 128, 1)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 128, 2)
    x = residual_block(x, 256, 2)
    x = residual_block(x, 512, 2)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, Y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(x_test, Y_test))

    return model, history


# In[32]:


model, history = ResNet34(x_train, Y_train, x_test, Y_test, num_classes=2, batch_size=32, epochs=50)


# In[33]:


# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, Y_test)

# Print the evaluation metrics
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')


# In[34]:


#RNN

def RNN(x_train, Y_train, x_test, Y_test, batch_size, epochs):
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, Y_test))

    return model, history

model, history = RNN(x_train, Y_train, x_test, Y_test, batch_size=32, epochs=20)


# In[35]:


#evaluating the model

loss, accuracy = model.evaluate(x_test, Y_test)
print("Test set loss:", loss)
print("Test set accuracy:", accuracy)


# In[36]:


x_train.shape


# In[52]:


# Reshape x_train
x_train = x_train.reshape((-1, 1, 128, 128, 1))

x_train.shape


# In[ ]:


# Define the input shape
input_shape = (None, 128, 128, 1)

# Define the number of classes
num_classes = 2

# Build the model
model = Sequential()

# Convolutional layers
model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'), input_shape=input_shape))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

# Flatten the output of the convolutional layers
model.add(TimeDistributed(Flatten()))

# Recurrent layers
model.add(LSTM(64, return_sequences=False))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(x_train, Y_train, batch_size=32, epochs=10, validation_split=0.2)


# In[ ]:




