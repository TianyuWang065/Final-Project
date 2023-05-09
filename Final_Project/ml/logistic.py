import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from utils import Utils
from sklearn.model_selection import GridSearchCV

from dataload.loaddata import datas, labels
print("Logistic")
filename = "logistic_voice"
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


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}


# 定义逻辑回归模型
model = LogisticRegression()

# 设置需要调整的参数及其可能取值范围
param_grid = {
    'penalty': ['l1', 'l2','elasticnet'],
    'C': [0.1, 0.01, 10,0.005]
}

# 定义GridSearchCV对象
grid_search = GridSearchCV(
    model,  # 模型
    param_grid,  # 参数空间
    cv=5,  # 交叉验证次数
    scoring='accuracy',  # 评分方法
    n_jobs=-1  # 并行处理数量
)

def start():
    results = []
    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)


    # 拟合模型
    grid_search.fit(X_train, y_train)
    # 输出最优参数组合

    # 构建最优模型
    best_model = LogisticRegression(**grid_search.best_params_)
    # best_model = LogisticRegression(C=0.3)#特征融合
    best_model.fit(X_train, y_train)

    # 在测试集上进行预测，并输出评价指标
    y_pred = best_model.predict(X_test)

    
    results.append(evaluate(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

    avg_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
start()