import librosa,os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from utils import Utils
from settings import N_MFCC
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import classification_report


class MetricsHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.val_losses = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_accuracy'))
        print("epoch",epoch)
    
    def draw(self):
        epoch = 19
        # 绘制损失和精确度的折线统计图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(range(1, epoch+2), self.losses, label='Train Loss')
        ax1.plot(range(1, epoch+2), self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend(loc='upper right')

        ax2.plot(range(1, epoch+2), self.accs, label='Train Accuracy')
        ax2.plot(range(1, epoch+2), self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend(loc='lower right')
        plt.show()



# 定义函数来提取MFCC特征
def extract_features(file_path):
    audio, sr = librosa.load(file_path,sr=44100)
    mfcc_features = librosa.feature.mfcc(audio, sr=sr,n_mfcc=N_MFCC)
    # print(mfcc_features) [[[],[],[],[]]]
    return mfcc_features

# 构建标签的one-hot编码
def one_hot_encode(labels):
    num_classes = len(set(labels))
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels

# 最大最小值归一化
def normalize_mfcc(mfcc_features):
    mfcc_features = np.array(mfcc_features)
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(mfcc_features, axis=0)
    min_vals = np.min(mfcc_features, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    normalized_mfcc = (mfcc_features - min_vals) / (max_vals - min_vals)
    return normalized_mfcc

# 加载数据并提取MFCC特征
X = []
y = []
mp3_path = ''
wav_path = ''
for i in range(1500):
    feature_num = 500
    if os.path.exists(mp3_path.format(i)):
        label = Utils().get_label(i)
        if label is not None:
            features = extract_features(mp3_path.format(i))
            for k in range(len(features[0])//feature_num):
                temp = []
                for mfcc in features:
                    origin_feature = [mfcc[index] for index in range(k*feature_num-feature_num,k*feature_num)]
                    normalize_feature = normalize_mfcc(origin_feature)
                    temp.append(normalize_feature)
                X.append(np.array(temp))
                y.append(label)
    elif os.path.exists(wav_path.format(i)):
        label = Utils().get_label(i)
        if label is not None:
            features = extract_features(wav_path.format(i))
            for k in range(len(features[0])//feature_num):
                temp = []
                for mfcc in features:
                    origin_feature = [mfcc[index] for index in range(k*feature_num-feature_num,k*feature_num)]
                    normalize_feature = normalize_mfcc(origin_feature)
                    temp.append(normalize_feature)
                X.append(np.array(temp))
                y.append(label)


# 将MFCC特征转换为Numpy数组
X = np.array(X)
y = np.array(y)
# print(X.shape)

# 对标签进行one-hot编码
y = one_hot_encode(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]),dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 训练模型
metrics_history = MetricsHistory()
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test),callbacks=[metrics_history])

# 在测试集上进行评估
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test loss:", loss)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_t = []
for y in y_test:
    if y[0]:
        y_t.append(0)
    elif y[1]:
        y_t.append(1)
    elif y[2]:
        y_t.append(2)
    elif y[3]:
        y_t.append(3)
print(classification_report(y_t,y_pred_classes))
print("auc",Utils().multiclass_roc_auc_score(y_t,y_pred_classes))

metrics_history.draw()