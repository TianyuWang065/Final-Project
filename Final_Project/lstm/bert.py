import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import classification_report

from utils import Utils

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


# 加载数据
data = pd.read_csv("")



# 预处理文本，将多个句子合并成一个文本，并用分隔符划分成句子列表
texts = [text.split("\n") for text in data["text"]]
texts = [" ".join(sentences) for sentences in texts]
# 将标签转换成one-hot编码
labels = pd.get_dummies(data["label"]).values
# 使用Bert tokenizer加载词典
vocab_size = 1000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)



# 将文本转换成数字序列
sequences = tokenizer.texts_to_sequences(texts)

# 对数字序列进行填充，使其长度一致
maxlen = 10
padded_sequences = pad_sequences(sequences, maxlen=maxlen)
# 
# 将数据集拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
# print(X_train[0])
# 定义模型
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=maxlen))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation="softmax"))
# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
# while True:
#     pass 
# 训练模型
metrics_history = MetricsHistory()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=32,callbacks=[metrics_history])

# 测试模型
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