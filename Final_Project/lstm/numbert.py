from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.model_selection import train_test_split
from settings import N_MFCC
import librosa,os
import numpy as np
from utils import Utils
import matplotlib.pyplot as plt
from keras.callbacks import Callback



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



# 最大最小值归一化
def normalize_mfcc(mfcc_features):
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(mfcc_features, axis=0)
    min_vals = np.min(mfcc_features, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    normalized_mfcc = (mfcc_features - min_vals) / (max_vals - min_vals)
    return normalized_mfcc

# 获取所有音频文件的MFCC特征和对应的标签
def get_data_and_labels():
    mp3_path = ''
    wav_path = ''
    data = []
    labels = []
    for i in range(1500):
        if os.path.exists(mp3_path.format(i)):
            # 加载音频文件
            y, sr = librosa.load(mp3_path.format(i),sr=44100)
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            label = Utils().get_label(i)
            if label is not None:
                temp = []
                for mfcc in mfccs:
                    temp.append([mfcc[i] for i in range(100)])
                mfccs = np.array(temp)
                print(mfccs.shape)
                data.append(mfccs)
                labels.append(label)
        elif os.path.exists(wav_path.format(i)):
            # 加载音频文件
            y, sr = librosa.load(wav_path.format(i),sr=44100)
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            label = Utils().get_label(i)
            if label is not None:
                temp = []
                for mfcc in mfccs:
                    temp.append([mfcc[i] for i in range(100)])
                mfccs = np.array(temp)
                print(mfccs.shape)
                data.append(mfccs)
                labels.append(label)
    
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# 构建标签的one-hot编码
def one_hot_encode(labels):
    num_classes = len(set(labels))
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels



# 获取MFCC特征和标签数据
data, labels = get_data_and_labels()
print("data shape:",data.shape)
# 对标签进行one-hot编码
encoded_labels = one_hot_encode(labels)


# 将数据集分为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

print(data.shape)
print(train_data.shape)
print(type(train_data))

# 构建LSTM神经网络模型
model = Sequential()
model.add(LSTM(64, input_shape=(train_data[1], train_data[2])))
model.add(Dense(units=4, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
metrics_history = MetricsHistory()
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=20, batch_size=32,callbacks=[metrics_history])

loss, accuracy = model.evaluate(test_data, test_labels)
print("Test Accuracy:", accuracy)
print("Test loss:", loss)
metrics_history.draw()
# 打印模型结构
model.summary()
