import librosa,os
import numpy as np
from settings import N_MFCC


# 最大最小值归一化
def normalize_mfcc(mfcc_features):
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(mfcc_features, axis=0)
    min_vals = np.min(mfcc_features, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    normalized_mfcc = (mfcc_features - min_vals) / (max_vals - min_vals)
    return normalized_mfcc


mp3_path = ''

mfccs_mean = []
mfccs_std = []

for i in range(100):
    if os.path.exists(mp3_path.format(i)):

        # 加载音频文件
        y, sr = librosa.load(mp3_path.format(i), sr=44100)

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # 获取MFCC特征的平均值
        mean = np.mean(mfccs, axis=1)

        # 获取MFCC特征的标准差
        std = np.std(mfccs, axis=1)

        mfccs_mean.append(normalize_mfcc(mean))
        mfccs_std.append(normalize_mfcc(std))

# 打印特征值
print('MFCCs mean:', mfccs_mean)
print('MFCCs std:', mfccs_std)
