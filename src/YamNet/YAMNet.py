import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import scipy
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, matthews_corrcoef, recall_score
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Function to read audio files and ensure sample rate
def read_audio(filename, desired_sample_rate=44100):
    wav_data, sample_rate = sf.read(filename, dtype=np.int16)
    print("filename,sample_rate",filename,sample_rate)

    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    if sample_rate != desired_sample_rate:
        wav_data = scipy.signal.resample(wav_data, int(len(wav_data) / sample_rate * desired_sample_rate))
    return wav_data

# Load YAMNet model from TensorFlow Hub
model_yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# Function to extract YAMNet embeddings from audio data
def extract_embeddings(audio_data):
    waveform = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    scores, embeddings, spectrogram = model_yamnet(waveform)
    return embeddings

# Function to preprocess data and extract embeddings
def preprocess_data(data_path, desired_sample_rate=44100, max_length=None):
    audio_data = []
    data = []
    class_labels = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            audio = read_audio(file_path, desired_sample_rate)
            data.append([audio,folder])
            embeddings = extract_embeddings(audio)
            # Pad or truncate embeddings to a fixed length
            if max_length is not None:
                embeddings = pad_or_truncate(embeddings, max_length)
            audio_data.append(embeddings)
            class_labels.append(folder)

    audio_dataframe = pd.DataFrame(data, columns=["audio_data", "class"])
    print(audio_dataframe)       
    return np.array(audio_data), np.array(class_labels)

def pad_or_truncate(embeddings, max_length):
    if embeddings.shape[0] < max_length:
        # Pad embeddings with zeros
        embeddings = np.pad(embeddings, ((0, max_length - embeddings.shape[0]), (0, 0)), mode='constant')
    elif embeddings.shape[0] > max_length:
        # Truncate embeddings
        embeddings = embeddings[:max_length, :]
    return embeddings

# Set data path and preprocess data
data_path = "./../train_audio"
max_length = 400  # Adjust as needed

flag = 0
if flag == 1:
    audio_data, class_labels = preprocess_data(data_path, max_length=max_length)
    np.savez("10_audio_data_and_labels.npz", audio_data=audio_data, class_labels=class_labels)
    print("Data saved successfully.")

# Load the saved data
loaded_data = np.load("10_audio_data_and_labels.npz")

# Access the loaded data
audio_data = loaded_data['audio_data']
class_labels = loaded_data['class_labels']

# Encode class labels
label_encoder = LabelEncoder()
class_labels_encoded = label_encoder.fit_transform(class_labels)
num_classes = len(label_encoder.classes_)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(audio_data, class_labels_encoded, test_size=0.1, random_state=42)

# Define and compile the model
model = models.Sequential([
    layers.Input(shape=(max_length, 1024)),
    
    # Convolutional layers
    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(256, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    # Global average pooling
    layers.GlobalAveragePooling1D(),
    
    # Dense layers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.1),  
    
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super(MetricsCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.precision_values = []
        self.recall_values = [] 
        self.f1_values = []
        self.mcc_values = []

    def on_epoch_end(self, epoch, logs=None):
        y_val_pred = self.model.predict(self.x_val)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        precision = precision_score(self.y_val, y_val_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(self.y_val, y_val_pred_classes, average='weighted', zero_division=0) 
        f1 = f1_score(self.y_val, y_val_pred_classes, average='weighted')
        mcc = matthews_corrcoef(self.y_val, y_val_pred_classes)
        print(f'Epoch {epoch + 1}: Precision - {precision:.4f}, Recall - {recall:.4f}, F1 Score - {f1:.4f}, MCC - {mcc:.4f}')
        self.precision_values.append(precision)
        self.recall_values.append(recall)
        self.f1_values.append(f1)
        self.mcc_values.append(mcc)
# Define MetricsCallback
metrics_callback = MetricsCallback(x_val, y_val)

# Train the model with early stopping and metrics callback
history = model.fit(x_train, y_train, epochs=25, batch_size=64, validation_data=(x_val, y_val), callbacks=[metrics_callback])


for epoch in range(len(history.history['accuracy'])):
    print("Epoch {}: Training Accuracy = {:.4f}, Training Loss = {:.4f}, Validation Accuracy = {:.4f}, Validation Loss = {:.4f}".format(
        epoch + 1, 
        history.history['accuracy'][epoch], 
        history.history['loss'][epoch], 
        history.history['val_accuracy'][epoch], 
        history.history['val_loss'][epoch]
    ))

accuracy = history.history['accuracy'].pop()
# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(f'{accuracy:4f}_accuracy_vs_epoch.png')
plt.show()

# Plot training and validation loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f'{accuracy:4f}_loss_vs_epoch.png')
plt.show()

# Plot precision, F1 score, and MCC on each epoch
epochs = range(1, len(metrics_callback.precision_values) + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs, metrics_callback.precision_values, label='Precision')
plt.plot(epochs, metrics_callback.recall_values, label='Recall')
plt.plot(epochs, metrics_callback.f1_values, label='F1 Score')
plt.plot(epochs, metrics_callback.mcc_values, label='MCC')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision, Recall, F1 Score, and MCC on Validation Data') 
plt.legend()
plt.savefig(f'{accuracy:4f}_precision_recall_f1_mcc_vs_epoch.png') 
plt.show()

# Save the model
model_path = f'audio_classification_model_v2_{accuracy:4f}.keras'
model.save(model_path)
print("Model saved successfully at", model_path)
