import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report
from utils import Utils
from dataload.loaddata import datas, labels

print("SVM")
filename = "svm_voice"
def load_local_data():
    
    with open(filename+"data.csv",mode="r",encoding="utf-8") as rf:
        csv_reader = csv.reader(rf)
        for item in csv_reader:
            temp = []
            for d in item:
                temp.append(float(d))
            datas.append(temp)
            
    
    with open(filename+"label.csv",mode="r",encoding="utf-8") as rf:
        csv_reader = csv.reader(rf)
        for item in csv_reader:
            for l in item:
                labels.append(int(l))


# 最大最小值归一化
for data in datas:
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(data, axis=0)
    min_vals = np.min(data, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    data = (data - min_vals) / (max_vals - min_vals)
    

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)

# 定义超参数范围
param_grid = {'C': [ 8],
              'gamma': [8],
              'kernel': ['poly']}

best_svm_model = svm.SVC(C=0.4)
best_svm_model.fit(X_train, y_train)

# 在测试集上进行预测和评估
y_pred = best_svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

Utils().save_data_label(datas,labels,filename)
