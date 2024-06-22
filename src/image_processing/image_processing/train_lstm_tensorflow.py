import cv2
import os
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_prefix

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

random_seed = 53

import time
start_time = time.time()

package_path = find_package_path('image_processing')
yolo_model = YOLO(os.path.join(package_path,'models','yolov8n-pose.pt'))
workspace = os.path.join(package_path, 'dataset_action_split')

def extract_keypoints(color_image):
    xyn_list = yolo_model.predict(color_image,
                verbose=False, # no print
                max_det=1,
                device="cuda:0")[0].keypoints.xyn.cpu().tolist()
    return [point for sublist in xyn_list[0] for point in sublist]

def create_whole_sequence(video_path):
    sequence = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            keypoints = extract_keypoints(frame)
            if len(keypoints) == 34:
                sequence.append(keypoints)
        else:
            break
    cap.release()
    return sequence

def create_regular_sequences(whole_sequence, seq_length=10, skip=2):
    sequences = []
    for i in range(0, len(whole_sequence) - seq_length + 1, skip):
        sequences.append(whole_sequence[i:i + seq_length])
    return sequences

def load_data(label_0, label_1, seq_length=10, skip=2):
    X, y = [], []
    for middle_path in label_0:
        each_path = os.path.join(workspace, middle_path)
        for video_name in os.listdir(each_path):
            video_path = os.path.join(each_path, video_name)
            whole_sequence = create_whole_sequence(video_path)
            regular_sequences = create_regular_sequences(whole_sequence, seq_length, skip)
            X.extend(regular_sequences)
            y.extend([0] * len(regular_sequences))
    seq_len_0 = len(y)
    print("seq_len_0:", seq_len_0)

    for middle_path in label_1:
        each_path = os.path.join(workspace, middle_path)
        for video_name in os.listdir(each_path):
            video_path = os.path.join(workspace, each_path, video_name)
            whole_sequence = create_whole_sequence(video_path)
            regular_sequences = create_regular_sequences(whole_sequence, seq_length, skip)
            X.extend(regular_sequences)
            y.extend([1] * len(regular_sequences))
    seq_len_1 = len(y) - seq_len_0
    print("seq_len_1:", seq_len_1)

    return np.array(X), np.array(y)

def build_lstm_model(num_features):
    model = Sequential()
    model.add(Input(shape=(None, num_features)))
    
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Fully connected layer
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Fully connected layer
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Fully connected layer
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

feature_path = os.path.join(package_path,'models','features.npz')
if os.path.exists(feature_path):
    data = np.load(feature_path)
    X = data['X']
    y = data['y']
else:
    train_label_0 = ['train/Sit down', 'train/Sitting', 'train/Lying Down', 'test/Sit down', 'test/Sitting', 'test/Lying Down']
    train_label_1 = ['train/Stand up', 'train/Standing', 'train/Walking', 'test/Stand up', 'test/Standing', 'test/Walking']
    X, y = load_data(train_label_0, train_label_1)
    np.savez_compressed(feature_path, X=X, y=y)

# Split data -> train : val : test = 3 : 1 : 1
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=random_seed)

# # Sizes of the splits
# print(f"Train size: {len(X_train)}")
# print(f"Validation size: {len(X_val)}")
# print(f"Test size: {len(X_test)}")

# Build and train the LSTM model
num_samples, seq_length, num_features = X.shape
model = build_lstm_model(num_features)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.0001)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=1024, callbacks=[early_stopping, reduce_lr], verbose=0)

train_accuracy = history.history['accuracy'][-1]
print(f'Last epoch train accuracy: {train_accuracy}')
val_accuracy = history.history['val_accuracy'][-1]
print(f'Last epoch validation accuracy: {val_accuracy}')

# 학습 과정에서 기록된 정확도와 손실 추출
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 첫 번째 subplot: 정확도
ax1.plot(epochs, train_acc, 'bo-', label='Training accuracy')  # 훈련 정확도 (파란색)
ax1.plot(epochs, val_acc, 'ro-', label='Validation accuracy')  # 검증 정확도 (빨간색)
ax1.set_title('Training and validation accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# 두 번째 subplot: 손실
ax2.plot(epochs, train_loss, 'bo-', label='Training loss')  # 훈련 손실 (파란색)
ax2.plot(epochs, val_loss, 'ro-', label='Validation loss')  # 검증 손실 (빨간색)
ax2.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
lstm_model_path = os.path.join(package_path,'models','lstm_model.keras')
plot_path = os.path.join(package_path,'models','confusion_matrix.png')
cm_path = os.path.join(package_path,'models','Training_and_validation.png')
if os.path.exists(lstm_model_path):
    os.remove(lstm_model_path)
if os.path.exists(plot_path):
    os.remove(plot_path)
if os.path.exists(cm_path):
    os.remove(cm_path)
plt.savefig(cm_path, dpi=400)

# 모델 저장 (Keras 기본 형식 사용)
save_model(model, lstm_model_path)

# 모델 불러오기
loaded_model = load_model(lstm_model_path)

# 모델 평가
y_pred_prob = loaded_model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['sit', 'stand'])
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix')
plt.savefig(plot_path, dpi=400)

# Precision, Recall, F1 Score 계산
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 결과 출력
print(f'Precision: {precision:.8f}')
print(f'Recall: {recall:.8f}')
print(f'F1 Score: {f1:.8f}')

# Evaluate the model
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f'Model accuracy: {accuracy * 100:.2f}%')

elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
