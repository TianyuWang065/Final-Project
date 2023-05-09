from sklearn.model_selection import GridSearchCV,train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from utils import Utils
from dataload.loaddata import datas, labels
print("Forest")
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy



# 定义保存评估结果的列表
rf_accuracy = []
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [150],
    'max_depth': [15],
    'min_samples_split': [4],
    'max_features':[len(datas[0])]
}

best_forest_model = RandomForestClassifier(max_depth=5,max_features=60,min_samples_split=4,n_estimators=100,random_state=42)
best_forest_model.fit(X_train, y_train)

y_pred = best_forest_model.predict(X_test)
# 评估模型并保存结果
acc = evaluate(y_test, y_pred)
rf_accuracy.append(acc)
print(classification_report(y_test,y_pred))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

