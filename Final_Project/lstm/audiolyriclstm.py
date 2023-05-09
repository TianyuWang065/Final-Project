import librosa
import csv
import os
import jieba
import numpy as np
from keras.layers import Input, Dense, concatenate, LSTM,Embedding
from keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from utils import Utils
from sklearn.metrics import classification_report
from settings import N_MFCC
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

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
        epoch = 4
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



class AudioLyricLSTM(object):


    def __init__(self,lyric_feature_dim,audio_feature_dim,show=False) -> None:
        self.lyric_feature_dim = lyric_feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.num_classes = 4
        self.merge_model = None
        self.X_lyric_train = []
        self.X_audio_train = []
        self.Y_label_train = []
        self.X_lyric_val = []
        self.X_audio_val =[]
        self.Y_label_val = []

        # 标注内容[{"music_id":1,"label":2}]
        self.annotation = []
        # 未划分的原数据文本
        self.origin_lyric_data = []
        # 原音频数据
        self.origin_audio_data = []
        # 处理好的文本特征集合
        self.lyric_data = []
        # 处理好的音频特征集合
        self.audio_data = []
        # 标签的集合
        self.labels = []
        # 
        self.audio_music_id = []


        self.load_annotation()
        self.load_audio()
        self.load_lyric()

        self.split_data()
        self.create_model()
        self.compile_fit()


        if show:
            self.__print()

    def split_data(self):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=4)  # 将数据集分成K份
        print(len(self.lyric_data))
        print(len(self.audio_data))
        print(len(self.labels))
        for train_index, test_index in kf.split(self.lyric_data):
            self.X_lyric_train, self.X_lyric_val = self.lyric_data[train_index], self.lyric_data[test_index]
            self.X_audio_train, self.X_audio_val = self.audio_data[train_index], self.audio_data[test_index]
            self.Y_label_train, self.Y_label_val = self.labels[train_index], self.labels[test_index]
      

    def load_lyric(self):
        for anno in self.annotation:
            if anno['music_id'] in self.audio_music_id:
                status,lyric_feature = self.__lyric_tf_idf(anno['music_id'])
                if not status:
                    continue
                self.lyric_data.append(lyric_feature)
                self.labels.append(anno['label'])
        self.lyric_data = np.array(self.lyric_data)
        self.labels = np.array(self.labels)

    def load_audio(self):
        # 加载数据并提取MFCC特征
        lyric_path = ""
        mp3_path = ''
        wav_path = ''
        for i in range(1500):
            feature_num = 300
            if os.path.exists(mp3_path.format(i)) and os.path.exists(lyric_path.format(i)):
                label = Utils().get_label(i)
                if label is not None:
                    features = self.__extract_features(mp3_path.format(i))
                    temp = []
                    for mfcc in features:
                        origin_feature = [mfcc[index] for index in range(feature_num)]
                        normalize_feature = self.__normalize_mfcc(origin_feature)
                        temp.append(normalize_feature)
                    self.audio_data.append(np.array(temp))
                    self.audio_music_id.append(i)
            elif os.path.exists(wav_path.format(i)) and os.path.exists(lyric_path.format(i)):
                label = Utils().get_label(i)
                if label is not None:
                    features = self.__extract_features(wav_path.format(i))
                    temp = []
                    for mfcc in features:
                        origin_feature = [mfcc[index] for index in range(feature_num)]
                        normalize_feature = self.__normalize_mfcc(origin_feature)
                        temp.append(normalize_feature)
                    self.audio_data.append(np.array(temp))
                    self.audio_music_id.append(i)



        # 将MFCC特征转换为Numpy数组
        self.audio_data = np.array(self.audio_data)


    def load_annotation(self):
        annotation_file = ""
        arf = open(annotation_file,mode="r",encoding="utf-8")
        annotation_reader = list(csv.DictReader(arf))

        for a in annotation_reader:
            music_id = int(a["musicId"])
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
            self.annotation.append({"label":label,"music_id":music_id})

    def create_model(self):
        # 定义模型的输入层
        lyric_input = Input(shape=(self.lyric_data.shape[1], self.lyric_data.shape[2]))
        audio_input = Input(shape=(self.audio_data.shape[1], self.audio_data.shape[2]))

        # 定义LSTM层
        lstm_lyric = LSTM(64)(lyric_input)
        lstm_audio = LSTM(64)(audio_input)

        # 合并LSTM层的输出
        merged = concatenate([lstm_lyric, lstm_audio])

        # 定义输出层
        output = Dense(4, activation='softmax')(merged)

        # 定义模型
        self.merge_model = Model(inputs=[lyric_input, audio_input], outputs=output)
        # print(self.merge_model.summary())

    def compile_fit(self):
        # 编译模型
        self.merge_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 定义EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        metrics_history = MetricsHistory()

        # 训练模型
        history = self.merge_model.fit(x=[self.X_lyric_train, self.X_audio_train], y=self.Y_label_train,  validation_data=([self.X_lyric_val, self.X_audio_val], self.Y_label_val), epochs=5, callbacks=[metrics_history])


        # 测试模型
        loss, accuracy = self.merge_model.evaluate(x=[self.X_lyric_val,self.X_audio_val], y=self.Y_label_val)
        print("Test Accuracy:", accuracy)
        print("Test loss:", loss)
        y_pred = self.merge_model.predict(x=[self.X_lyric_val,self.X_audio_val])
        y_pred_classes = y_pred.argmax(axis=-1)
        label_encoder = LabelEncoder()
        label_encoder.fit(self.Y_label_val)
        y_pred = label_encoder.inverse_transform(y_pred_classes)
        print(classification_report(self.Y_label_val,y_pred))
        print("auc",Utils().multiclass_roc_auc_score(self.Y_label_val,y_pred))
        metrics_history.draw()

    def __lyric_tf_idf(self,music_id):
        file_path = ""
        X = None
        path = "{}{}.lrc".format(file_path,music_id)
        if not os.path.exists(path):
            return False,None
        with open(path, "r") as rf:
            data = [line.replace("\n", "") for line in rf.readlines()]
        data_len = len(data)
        lis = []
        s1 = ""
        part = 5
        for temp in data[0:data_len//part]:
            s1 += self.__cut_word(temp)

        s2 = ""
        for temp in data[data_len//part:data_len*2//part]:
            s2 += self.__cut_word(temp)

        s3 = ""
        for temp in data[data_len*2//part:data_len*3//part]:
            s3 += self.__cut_word(temp)
        
        s4 = ""
        for temp in data[data_len*3//part:]:
            s4 += self.__cut_word(temp)

        lis.append(s1)
        lis.append(s2)
        lis.append(s3)
        lis.append(s4)
        transfer = TfidfVectorizer(max_features=self.lyric_feature_dim)
        try:
            X = transfer.fit_transform(lis).toarray()
        except:
            return False ,None 
        Z = []
        for x in X:
            x.astype(float)
            z = np.pad(x,(0,self.lyric_feature_dim-len(x)),'constant',constant_values=(0))
            Z.append(z)
        Z = np.array(Z)        
        return True,Z

    def __cut_word(self,data):
        return " ".join(list(jieba.cut(data)))

 
    def __print(self):
        print("self.lyric_data length:",len(self.lyric_data))
        print("self.audio_data length:",len(self.audio_data))
        print("self.labels length:",len(self.labels))

        print("lyric_data shape",self.lyric_data.shape)
        print("audio_data shape",self.audio_data.shape)

        # print(self.lyric_data[0])

    # 定义函数来提取MFCC特征
    def __extract_features(self,file_path):
        audio, sr = librosa.load(file_path,sr=44100)
        mfcc_features = librosa.feature.mfcc(audio, sr=sr,n_mfcc=N_MFCC)
        return mfcc_features
    
    # 最大最小值归一化
    def __normalize_mfcc(self,mfcc_features):
        mfcc_features = np.array(mfcc_features)
        # 沿着第二个轴计算每个特征维度上的最大值和最小值
        max_vals = np.max(mfcc_features, axis=0)
        min_vals = np.min(mfcc_features, axis=0)
        # 对每个特征维度上的特征值进行归一化处理
        normalized_mfcc = (mfcc_features - min_vals) / (max_vals - min_vals)
        return normalized_mfcc

