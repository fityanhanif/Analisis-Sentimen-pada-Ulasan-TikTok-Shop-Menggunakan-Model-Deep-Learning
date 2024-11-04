from google.colab import files
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# -*- coding: utf-8 -*-

uploaded = files.upload()

#Cek Data
data = pd.read_csv("Tiktok Tokopedia Seller Center Reviews.csv", delimiter=",")
data.tail(10)

# Membuat DataFrame
df = pd.DataFrame(data)

# Mengubah nilai sentimen menjadi 0 dan 1
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Menampilkan DataFrame yang telah diubah
print(df)

# Menghitung jumlah positif dan negatif
sentiment_counts = df['sentiment'].value_counts().rename(index={0: 'Negative', 1: 'Positive'})

# Membuat grafik batang
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Jumlah Sentimen Positif dan Negatif')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Menampilkan grafik
plt.show()

pip install wordcloud matplotlib

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Menampilkan beberapa baris dari DataFrame untuk memastikan data terbaca dengan baik
print(df.head())

# Misalkan kolom yang berisi teks ulasan bernama 'review'
# Ganti 'review' dengan nama kolom yang sesuai jika berbeda
text_combined = ' '.join(df['content'])

# Membuat word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

# Menampilkan word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Menghilangkan sumbu
plt.title('Word Cloud dari Ulasan Seller Center')
plt.show()

nltk.download('punkt')

# Jika belum, Anda mungkin perlu mengunduh resource berikut:
# nltk.download('punkt')

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Menggabungkan semua teks menjadi satu string
text_combined = ' '.join(df['content'])

# Tokenisasi
tokens = word_tokenize(text_combined)

# Menampilkan beberapa token
print(tokens[:10])  # Menampilkan 10 token pertama

pip install tensorflow

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Menampilkan beberapa baris dari DataFrame untuk memastikan data terbaca dengan baik
print(df.head())

# Misalkan kolom yang berisi teks ulasan bernama 'review'
# Ganti 'review' dengan nama kolom yang sesuai jika berbeda
texts = df['content'].astype(str).tolist()  # Mengubah kolom menjadi list string

# Membuat tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Mengubah teks menjadi urutan angka
sequences = tokenizer.texts_to_sequences(texts)

# Menentukan panjang maksimum untuk padding
max_length = max(len(seq) for seq in sequences)

# Menambahkan padding
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Menampilkan hasil padding
print("\nHasil Padding:")
print(padded_sequences[:5])  # Menampilkan 5 urutan pertama setelah padding

pip install scikit-learn

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Menampilkan beberapa baris dari DataFrame untuk memastikan data terbaca dengan baik
print(df.head())

# Misalkan kolom yang berisi teks ulasan bernama 'review' dan kolom sentimen 'sentiment'
# Ganti dengan nama kolom yang sesuai jika berbeda
texts = df['content'].astype(str).tolist()
labels = df['sentiment'].tolist()

# Membagi data menjadi 80% untuk pelatihan dan 20% untuk pengujian
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Menampilkan hasil pembagian
print(f'\nJumlah data pelatihan: {len(X_train)}')
print(f'Jumlah data pengujian: {len(X_test)}')

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Misalkan kolom yang berisi teks ulasan bernama 'review' dan kolom sentimen 'sentiment'
texts = df['content'].astype(str).tolist()
labels = df['sentiment'].tolist()

# Mengubah label ke format numerik
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Membagi data menjadi 80% untuk pelatihan dan 20% untuk pengujian
X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

# Membuat tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Mengubah teks menjadi urutan angka
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Menentukan panjang maksimum untuk padding
max_length = max(len(seq) for seq in X_train_seq)

# Menambahkan padding
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Mengonversi label ke kategori
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Membangun model LSTM
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Ganti 2 jika ada lebih dari 2 kelas

# Mengompilasi model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
history = model.fit(X_train_padded, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_test_padded, y_test_categorical))

# Menampilkan hasil pelatihan
loss, accuracy = model.evaluate(X_test_padded, y_test_categorical)
print(f'\nLoss: {loss}')
print(f'Accuracy: {accuracy}')

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Misalkan kolom yang berisi teks ulasan bernama 'review' dan kolom sentimen 'sentiment'
texts = df['content'].astype(str).tolist()
labels = df['sentiment'].tolist()

# Mengubah label ke format numerik
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Membagi data menjadi 80% untuk pelatihan dan 20% untuk pengujian
X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

# Membuat tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Mengubah teks menjadi urutan angka
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Menentukan panjang maksimum untuk padding
max_length = max(len(seq) for seq in X_train_seq)

# Menambahkan padding
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Pastikan y_train dan y_test memiliki ukuran yang sesuai
print(f'Jumlah X_train_padded: {len(X_train_padded)}')
print(f'Jumlah y_train: {len(y_train)}')
print(f'Jumlah X_test_padded: {len(X_test_padded)}')
print(f'Jumlah y_test: {len(y_test)}')

# Membangun model LSTM
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Satu neuron output untuk binary classification

# Mengompilasi model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih

# 1. Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# 2. Menyiapkan teks dan label
texts = df['content'].astype(str).tolist()  # Mengambil kolom ulasan
labels = df['sentiment'].tolist()           # Mengambil kolom sentimen

# 3. Mengubah label ke format numerik
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 4. Membagi data menjadi 80% untuk pelatihan dan 20% untuk pengujian
X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

# 5. Membuat tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 6. Mengubah teks menjadi urutan angka
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 7. Menentukan panjang maksimum untuk padding
max_length = max(len(seq) for seq in X_train_seq)

# 8. Menambahkan padding
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# 9. Membangun model LSTM
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Satu neuron output untuk binary classification

# 10. Mengompilasi model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 11. Melatih model
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_data=(X_test_padded, y_test))

# 12. Menampilkan hasil pelatihan
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'\nLoss: {loss}')
print(f'Accuracy: {accuracy}')

pip install seaborn matplotlib

# Melakukan prediksi pada data pengujian
y_pred_probs = model.predict(X_test_padded)
y_pred = (y_pred_probs > 0.5).astype(int)  # Menggunakan threshold 0.5 untuk klasifikasi biner

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Menampilkan confusion matrix dengan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# Grafik Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (Training)')
plt.plot(history.history['val_loss'], label='Loss (Validation)')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Grafik Akurasi
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy (Training)')
plt.plot(history.history['val_accuracy'], label='Accuracy (Validation)')
plt.title('Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Fungsi untuk memprediksi sentimen dari kalimat baru
def predict_sentiment(text):
    # Tokenisasi kalimat baru
    seq = tokenizer.texts_to_sequences([text])

    # Padding untuk memastikan panjang input sesuai dengan model
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    # Melakukan prediksi
    prediction_prob = model.predict(padded)
    prediction = (prediction_prob > 0.5).astype(int)  # Menggunakan threshold 0.5

    return prediction[0][0], prediction_prob[0][0]

# Contoh kalimat untuk diprediksi
sample_sentences = [
    "Kualitasnya jelek dan lama sampai.",
]

# Melakukan prediksi untuk setiap kalimat contoh
for sentence in sample_sentences:
    sentiment, probability = predict_sentiment(sentence)
    sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
    print(f"Kalimat: '{sentence}' - Sentimen: {sentiment_label}, Probabilitas: {probability:.4f}")

!pip install wordcloud matplotlib

# Membaca data dari file CSV
file_path = 'Tiktok Tokopedia Seller Center Reviews.csv'
df = pd.read_csv(file_path)

# Memisahkan ulasan positif dan negatif
positive_reviews = df[df['sentiment'] == 'positive']['content']
negative_reviews = df[df['sentiment'] == 'negative']['content']

# Menggabungkan semua ulasan positif dan negatif menjadi satu string
positive_text = ' '.join(positive_reviews.astype(str).tolist())
negative_text = ' '.join(negative_reviews.astype(str).tolist())

# Create a word cloud for negative sentiment
wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

# Create a word cloud for positive sentiment
wordcloud_positive = WordCloud(width=800, height=400, background_color='black', colormap='Blues').generate(positive_text)

# Menampilkan word cloud positif
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positif')

# Menampilkan word cloud negatif
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negatif')

plt.tight_layout()
plt.show()