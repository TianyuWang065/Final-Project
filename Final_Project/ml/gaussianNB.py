import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from utils import Utils
from dataload.loaddata import datas, labels
print("GaussianNB")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)

param_grid = {'var_smoothing': [1e-9, 1e-10, 1e-11, 1e-12, 1e-15]}

# 定义随机森林分类器
gnb_model = GaussianNB()

#定义网格搜索对象
grid_search = GridSearchCV(gnb_model, param_grid, cv=5,scoring='accuracy')

# 进行交叉验证和参数调整
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("Best parameters: {}".format(grid_search.best_params_))

# 定义GaussianNB分类器
gnb = GaussianNB(var_smoothing=0.07)

# 训练模型
gnb.fit(X_train, y_train)

# 在测试集上进行预测和评估
y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))