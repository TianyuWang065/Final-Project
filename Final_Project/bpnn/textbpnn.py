import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from utils import Utils



TEXT_FEATURE_NUM = 200


def lyric_tf_idf(text):
    X = None
    transfer = TfidfVectorizer(max_features=TEXT_FEATURE_NUM)
    X = transfer.fit_transform([text]).toarray()
    for item in X:
        item.astype(float)
        item = np.pad(item,(0,TEXT_FEATURE_NUM-len(item)),'constant',constant_values=(0))
        return item
    return None


# 加载数据
data = pd.read_csv("")
lyric_data = data["text"] # 歌词文本数据
y = data["label"] # 标签

# 数据预处理
# 将情感标签转换成数字编码
le = LabelEncoder()
y = le.fit_transform(y)
# 将文本数据转换成数值向量
X = []
for text in lyric_data:
    X.append(lyric_tf_idf(text))



# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 构建BP神经网络模型
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True)

# 定义损失列表
losses = []

# 训练模型
for epoch in range(1000):
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    loss = model.loss_
    losses.append(loss)
    print("Epoch {}: Loss = {}".format(epoch, loss))

# 绘制损失曲线
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# 训练模型
# model.fit(X_train, y_train)

# 测试模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# 打印结果
print("Accuracy: {:.2f}%".format(accuracy*100))
print(classification_report(y_test,y_pred))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))
