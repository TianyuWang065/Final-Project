import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba,os,csv
from settings import RANDOM_PICK_FEATURE_NUM,TEXT_FEATURE_NUM
from utils import Utils
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from keras.callbacks import Callback
EPOCH = 10
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
        epoch = EPOCH-1
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
lyric_data = []
audio_data = []
labels = []



# 最大最小值归一化
def normalize_mfcc(mfcc_features):
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(mfcc_features, axis=0)
    min_vals = np.min(mfcc_features, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    normalized_mfcc = (mfcc_features - min_vals) / (max_vals - min_vals)
    return normalized_mfcc


static_feature_file_path = ""

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
            'data':list(normalize_mfcc(np.array(temp_data)))
        })


file_path = ""


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

annotation = ""
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
            for feature in feature_data:
                if feature['musicId'] == music_id:
                    lyric_data.append(list(item))
                    audio_data.append(feature['data'])
                    labels.append(lable)
arf.close()

lyric_data = np.array(lyric_data)
audio_data = np.array(audio_data)
labels = np.array(labels)


# while True:
#     pass 
# 划分训练集和测试集
X_lyric_train, X_lyric_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(lyric_data, audio_data, labels, test_size=0.2, random_state=42)

# 创建BP神经网络模型


lyric_input = Input(shape=(X_lyric_train.shape[1],))
audio_input = Input(shape=(X_audio_train.shape[1],))
concatenated = Concatenate()([lyric_input, audio_input])
dense1 = Dense(200, activation='relu')(concatenated)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(4, activation='sigmoid')(dense2)
model = Model(inputs=[lyric_input, audio_input], outputs=output)


# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history_metrics = MetricsHistory()
# 训练模型
history = model.fit(x=[X_lyric_train, X_audio_train], y=y_train, validation_split=0.2, epochs=EPOCH,batch_size=32,verbose=2,callbacks=[history_metrics])

# 在测试集上进行预测
y_pred = model.predict([X_lyric_test, X_audio_test])
y_pred_classes = np.argmax(y_pred, axis=1)


print(classification_report(y_test,y_pred_classes))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred_classes))

history_metrics.draw()
