import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from dataload.randompickd import datas,labels
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from utils import Utils

# 加载数据
audio_data = datas
labels = labels

# 标准化数据
scaler = StandardScaler()
audio_data = scaler.fit_transform(audio_data)

# 划分训练集和测试集
train_samples = int(len(audio_data) * 0.8)
x_train = audio_data[:train_samples]
y_train = labels[:train_samples]
x_test = audio_data[train_samples:]
y_test = labels[train_samples:]

# 定义神经网络模型
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

# 定义损失列表
losses = []

# 训练模型
for epoch in range(1000):
    model.partial_fit(x_train, y_train, classes=np.unique(y_train))
    loss = model.loss_
    losses.append(loss)
    print("Epoch {}: Loss = {}".format(epoch, loss))

# 绘制损失曲线
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# 训练模型
# model.fit(x_train, y_train)

# 预测测试集数据
y_pred = model.predict(x_test)

# 输出模型评估结果
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)
print(classification_report(y_test,y_pred))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))
