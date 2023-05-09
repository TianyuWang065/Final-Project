from sklearn.feature_extraction.text import TfidfVectorizer
import jieba,os,csv
import numpy as np
from settings import RANDOM_PICK_FEATURE_NUM, TEXT_FEATURE_NUM
from utils import Utils

# 最大最小值归一化
def normalize_mfcc(mfcc_features):
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(mfcc_features, axis=0)
    min_vals = np.min(mfcc_features, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    normalized_mfcc = (mfcc_features - min_vals) / (max_vals - min_vals)
    return normalized_mfcc


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
            # 'data':list(normalize_mfcc(np.array(temp_data)))
            'data':temp_data
        })


file_path = "C:\\Users\\Tianyu Wang\\Desktop\\music-emotion\\clyrics\\"



def cut_word(sent):
    return " ".join(list(jieba.cut(sent)))


def lyric_tf_idf(i):
    X = None
    path = "{}{}.lrc".format(file_path,i)
    if os.path.exists(path):
        with open(path, "r") as rf:
            data = [line.replace("\n", "") for line in rf.readlines()]
        data_len = len(data)
        lis = []
        s1 = ""
        for temp in data[0:data_len]:
            s1 += cut_word(temp)
        lis.append(s1)
        
        transfer = TfidfVectorizer(max_features=TEXT_FEATURE_NUM)
        X = transfer.fit_transform(lis).toarray()
    return X 


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
            item = np.pad(item,(0,TEXT_FEATURE_NUM-len(item)),'mean')
            for feature in feature_data:
                if feature['musicId'] == music_id:
                    datas.append(list(item)+feature['data'])
                    labels.append(lable)
data_file_path = "fusiondata.csv"

for data in datas:
    data=normalize_mfcc(data)

with open(data_file_path,mode="w",encoding="utf-8") as wf:
    csv_wf = csv.writer(wf)
    for i in range(len(labels)):
        data = datas[i]
        data.append(labels[i]+1)
        csv_wf.writerow(data)

datas = np.array(datas)
labels = np.array(labels)

arf.close()