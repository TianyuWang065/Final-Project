from settings import RANDOM_PICK_FEATURE_NUM
import pandas as pd
import csv
import numpy as np
from utils import Utils

labels = []
datas = []

print("random pick")

annotation = "C:\\Users\\Tianyu Wang\\Desktop\\music-emotion\\newannotations\\thayer_static_annotations.csv"
arf = open(annotation,mode="r",encoding="utf-8")
annotation_reader = list(csv.DictReader(arf))
static_feature_file_path = "C:\\Users\\Tianyu Wang\\Desktop\\music-emotion\\features\\static_features.csv"

# 加载特征
feature_data = []
with open(static_feature_file_path,mode="r",encoding="utf-8") as rf:
    csv_reader = csv.DictReader(rf)
    field_len = len(csv_reader.fieldnames)
    pick_field = []
    while len(pick_field) != RANDOM_PICK_FEATURE_NUM:
        index = Utils.random_int(0,field_len-1)
        if csv_reader.fieldnames[index] not in pick_field:
            pick_field.append(csv_reader.fieldnames[index])
    for data in csv_reader:
        temp_data = []
        for key in pick_field:
            temp_data.append(float(data[key]))
        feature_data.append({
            'musicId':int(data['musicId']),
            'data':temp_data
        })

# 添加标注
for a in annotation_reader:
    music_id= int(a["musicId"])
    label = a["Thayer"].strip("]").strip("[").split(",")
    label[0] = int(label[0])
    label[1] = int(label[1])

    if label[0] == 1 and label[1] == 1:
        label = 0
    elif label[0] == -1 and label[1] == 1:
        label = 1
    elif label[0] == -1 and label[1] == -1:
        label = 2
    else:
        label = 3
    for feature in feature_data:
        if feature['musicId'] == music_id:
            datas.append(feature['data'])
            break
    labels.append(label)
    

datas = np.array(datas)
labels = np.array(labels)
arf.close()

