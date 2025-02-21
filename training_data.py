import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 🟢 Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv("data_training.csv")

# 🟢 Bước 2: Tách dữ liệu và nhãn (X, y)
X = df.drop(columns=["label"]).values
y = df["label"].values

# 🟢 Bước 3: Chia tập train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟢 Bước 4: Chuyển nhãn từ chữ sang số
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# 🟢 Bước 5: Xây dựng mô hình CNN
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),  
    layers.Dense(64, activation="relu"),
    layers.Dense(len(set(y_train_encoded)), activation="softmax")  
])

# Compile mô hình
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 🟢 Bước 6: Huấn luyện mô hình
model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test, y_test_encoded))

# 🟢 Bước 7: Đánh giá mô hình
loss, acc = model.evaluate(X_test, y_test_encoded)
print(f"🎯 Độ chính xác trên tập test: {acc * 100:.2f}%")

# 🟢 Bước 8: Dự đoán với dữ liệu mới
sample_index = 5
new_sample = X_test[sample_index].reshape(1, -1)
predicted_label = model.predict(new_sample)
predicted_class = encoder.inverse_transform([np.argmax(predicted_label)])

print(f"📝 Bài tập dự đoán: {predicted_class[0]}")
print(f"✅ Bài tập thực tế: {encoder.inverse_transform([y_test_encoded[sample_index]])[0]}")

# 🟢 Bước 9: Lưu mô hình đã train
model.save("exercise_cnn_model.h5")
print("💾 Mô hình đã được lưu thành công!")
