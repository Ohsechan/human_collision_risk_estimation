
import cv2
import os
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_prefix

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

random_seed = 53

package_path = find_package_path('image_processing')
yolo_model = YOLO(os.path.join(package_path, 'models', 'yolov8n-pose.pt'))
dataset_path = os.path.join(package_path, 'dataset_action_split')

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

def create_regular_sequences(whole_sequence, seq_length=5, skip=2):
    sequences = []
    for i in range(0, len(whole_sequence) - seq_length + 1, skip):
        sequences.append(whole_sequence[i:i + seq_length])
    return sequences

def load_data(label_0, label_1, seq_length=5, skip=2):
    X, y = [], []
    for middle_path in label_0:
        each_path = os.path.join(dataset_path, middle_path)
        for video_name in os.listdir(each_path):
            video_path = os.path.join(each_path, video_name)
            whole_sequence = create_whole_sequence(video_path)
            regular_sequences = create_regular_sequences(whole_sequence, seq_length, skip)
            X.extend(regular_sequences)
            y.extend([0] * len(regular_sequences))
    seq_len_0 = len(y)
    print("seq_len_0:", seq_len_0)

    for middle_path in label_1:
        each_path = os.path.join(dataset_path, middle_path)
        for video_name in os.listdir(each_path):
            video_path = os.path.join(each_path, video_name)
            whole_sequence = create_whole_sequence(video_path)
            regular_sequences = create_regular_sequences(whole_sequence, seq_length, skip)
            X.extend(regular_sequences)
            y.extend([1] * len(regular_sequences))
    seq_len_1 = len(y) - seq_len_0
    print("seq_len_1:", seq_len_1)

    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, num_features = 34):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_features, 64, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 16)
        self.batch_norm4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of LSTM

        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        x = F.relu(self.fc3(x))
        x = self.batch_norm4(x)
        x = self.dropout4(x)

        x = torch.sigmoid(self.fc4(x))
        return x

# 예측 및 평가
def evaluate_model(model, X_test, y_test, plot_path):
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test)
        y_pred = (y_pred_prob > 0.5).float()  # 예측값은 0 또는 1로 변환
        y_pred = y_pred.squeeze().cpu().numpy()
        y_test = y_test.numpy()

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
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
        accuracy = (y_pred == y_test).mean()
        print(f'Test accuracy: {accuracy * 100:.2f}%')

def main():
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
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model, optimizer, and loss function
    input_size = 34
    model = LSTMModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    # Training loop
    num_epochs = 2000
    batch_size = 1024
    best_val_acc = 0.0
    best_model_state = None

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train.float())
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())
        
        # Calculate training accuracy
        train_preds = torch.round(outputs).squeeze()
        train_acc = accuracy_score(y_train.numpy(), train_preds.detach().numpy())
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val.float()).item()
            val_loss_history.append(val_loss)
            
            # Calculate validation accuracy
            val_preds = torch.round(val_outputs).squeeze()
            val_acc = accuracy_score(y_val.numpy(), val_preds.detach().numpy())
            val_acc_history.append(val_acc)
            
            # Save the best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

        # Print progress (optional)
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_state)

    # Evaluate the best model on train set
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        train_preds = torch.round(train_outputs).squeeze()
        train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())

    # Print train accuracy, best validation accuracy, and test accuracy
    print(f"Train Accuracy of the Best Model: {train_acc * 100:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")

    # Plotting
    epochs = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # First subplot: Accuracy
    ax1.plot(epochs, train_acc_history, 'bo-', label='Training accuracy')  # Training accuracy (blue)
    ax1.plot(epochs, val_acc_history, 'ro-', label='Validation accuracy')  # Validation accuracy (red)
    ax1.set_title('Training and validation accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Loss
    ax2.plot(epochs, train_loss_history, 'bo-', label='Training loss')  # Training loss (blue)
    ax2.plot(epochs, val_loss_history, 'ro-', label='Validation loss')  # Validation loss (red)
    ax2.set_title('Training and validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    lstm_model_path = os.path.join(package_path,'models','lstm_model.pth')
    plot_path = os.path.join(package_path,'models','confusion_matrix.png')
    cm_path = os.path.join(package_path,'models','Training_and_validation.png')
    if os.path.exists(lstm_model_path):
        os.remove(lstm_model_path)
    if os.path.exists(plot_path):
        os.remove(plot_path)
    if os.path.exists(cm_path):
        os.remove(cm_path)
    plt.savefig(cm_path, dpi=400)

    torch.save(model.state_dict(), lstm_model_path)

    model = LSTMModel(input_size)
    model.load_state_dict(torch.load(lstm_model_path))

    # 모델 평가
    evaluate_model(model, X_test, y_test, plot_path)

if __name__ == '__main__':
    main()

