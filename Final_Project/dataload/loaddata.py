import csv
import numpy as np
from .tfidf import TEXT_FEATURE_NUM, lyric_tf_idf

labels = []
datas = []


annotation = "C:\\Users\\Tianyu Wang\\Desktop\\music-emotion\\newannotations\\thayer_static_annotations.csv"
arf = open(annotation,mode="r",encoding="utf-8")
annotation_reader = list(csv.DictReader(arf))

# 添加标注
for a in annotation_reader:
    music_id= int(a["musicId"])
    try:
        X = lyric_tf_idf(music_id)
    except:
        X = None 
    if X is not None:
        lable = a["Thayer"].strip("]").strip("[").split(",")
        lable[0] = int(lable[0])
        lable[1] = int(lable[1])

        if lable[0] == 1 and lable[1] == 1:
            lable = 0
        elif lable[0] == -1 and lable[1] == 1:
            lable = 1
        elif lable[0] == -1 and lable[1] == -1:
            lable = 2
        else:
            lable = 3
        for item in X:
            item.astype(float)
            item = np.pad(item,(0,TEXT_FEATURE_NUM-len(item)),'constant')
            datas.append(item)
            labels.append(lable)
train_datas = np.array(datas)
train_labels = np.array(labels)
arf.close()



labels = []
datas = []


annotation = "C:\\Users\\Tianyu Wang\\Desktop\\music-emotion\\newannotations\\thayer_static_annotations.csv"
arf = open(annotation,mode="r",encoding="utf-8")
annotation_reader = list(csv.DictReader(arf))

# 添加标注
for a in annotation_reader:
    music_id= int(a["musicId"])
    X = lyric_tf_idf(music_id)
    if X is not None:
        lable = a["Thayer"].strip("]").strip("[").split(",")
        lable[0] = int(lable[0])
        lable[1] = int(lable[1])

        if lable[0] == 1 and lable[1] == 1:
            lable = 0
        elif lable[0] == -1 and lable[1] == 1:
            lable = 1
        elif lable[0] == -1 and lable[1] == -1:
            lable = 2
        else:
            lable = 3
        for item in X:
            item.astype(float)
            item = np.pad(item,(0,TEXT_FEATURE_NUM-len(item)),'constant')
            datas.append(item)
            labels.append(lable)
train_datas = np.array(datas)
train_labels = np.array(labels)
arf.close()


