import tensorflow as tf
print("TensorFlow version:", tf.__version__)

!pip install ipython-autotime
%load_ext autotime

import tensorflow as tf
from google.colab import drive
import matplotlib.pyplot as plt
import keras
import re
import os
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load and preprocess dataset
file_path = '/content/drive/MyDrive/Genre Classification Dataset/test_data_solution.txt'
with open(file_path, 'r') as files:
    content = files.read()

lines = content.strip().split('\n')

titles, genres, descriptions = [], [], []
for line in lines:
    parts = line.split(' ::: ')
    if len(parts) == 4:
        movie_id, title, genre, description = parts
        titles.append(title.strip())
        genres.append(genre.strip())
        descriptions.append(description.strip())

df = pd.DataFrame({
    'title': titles,
    'genre': genres,
    'description': descriptions
})

csv_path = '/content/drive/MyDrive/Genre Classification Dataset/test_data_solution.txt.csv'
df.to_csv(csv_path, index=False)
df = pd.read_csv(csv_path)

X = df['description']
Y = df['genre']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)
Y = LabelEncoder().fit_transform(Y)

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Text preprocessing
stopwords = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'to', 'from', 'of', 'on', 'off',
    'over', 'under', 'than', 'so', 'such', 'as', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can',
    'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'
])
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = ' '.join(word for word in text.split() if word not in stopwords)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

X_train = [clean_text(text) for text in X_train]
X_val = [clean_text(text) for text in X_val]

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200, padding='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=200, padding='post')

# Neural Network Model 1
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(Y)), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training model 1
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# Neural Network Model 2
model2 = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(200, return_sequences=True, dropout=0.4)),
    Bidirectional(LSTM(200, dropout=0.4)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(len(np.unique(Y)), activation='softmax')
])
model2.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()

# Training model 2
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history2 = model2.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# SVM Model
with tf.device('/device:GPU:0'):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, Y_train)

# RandomForest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Evaluation on test data
test_dataset_path = '/content/drive/MyDrive/Genre Classification Dataset/test_data_solution.txt.csv'
test_df = pd.read_csv(test_dataset_path)
X_test = test_df['description']
Y_test = LabelEncoder().fit_transform(test_df['genre'])

# Preprocess test data
X_test = [clean_text(text) for text in X_test]
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200, padding='post')

# Neural Network Model Evaluation
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Model 1 - Loss: {loss}, Accuracy: {accuracy}")
loss, accuracy = model2.evaluate(X_test, Y_test)
print(f"Model 2 - Loss: {loss}, Accuracy: {accuracy}")

# SVM Evaluation
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(Y_test, y_pred_svm)
report_svm = classification_report(Y_test, y_pred_svm)
print(f"SVM - Accuracy: {accuracy_svm}")
print("SVM Classification Report:\n", report_svm)

# RandomForest Evaluation
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(Y_test, y_pred_rf)
report_rf = classification_report(Y_test, y_pred_rf)
print(f"Random Forest - Accuracy: {accuracy_rf}")
print("Random Forest Classification Report:\n", report_rf)

# Visualization of predicted genre counts
def plot_genre_counts(predicted_genres, title):
    genre_counts = Counter(predicted_genres)
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(genres, counts, color='skyblue')
    plt.title(title)
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.show()

predicted_classes_nn1 = np.argmax(model.predict(X_test), axis=1)
predicted_genres_nn1 = LabelEncoder().inverse_transform(predicted_classes_nn1)
plot_genre_counts(predicted_genres_nn1, 'Predicted Genre Counts - NN Model 1')

predicted_classes_nn2 = np.argmax(model2.predict(X_test), axis=1)
predicted_genres_nn2 = LabelEncoder().inverse_transform(predicted_classes_nn2)
plot_genre_counts(predicted_genres_nn2, 'Predicted Genre Counts - NN Model 2')

plot_genre_counts(y_pred_svm, 'Predicted Genre Counts - SVM Model')
plot_genre_counts(y_pred_rf, 'Predicted Genre Counts - Random Forest Model')

# Confusion Matrix for NN Model 1
cm = confusion_matrix(Y_test, predicted_genres_nn1)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LabelEncoder().classes_)
fig, ax = plt.subplots(figsize=(35, 30))
display.plot(cmap=plt.cm.Blues, ax=ax)
plt.show()

# Confusion Matrix for NN Model 2
cm = confusion_matrix(Y_test, predicted_genres_nn2)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LabelEncoder().classes_)
fig, ax = plt.subplots(figsize=(35, 30))
display.plot(cmap=plt.cm.Blues, ax=ax)
plt.show()
