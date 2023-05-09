from sklearn.feature_extraction.text import TfidfVectorizer
import jieba,os,csv


file_path = "C:\\Users\\Tianyu Wang\\Desktop\\music-emotion\\clyrics"

TEXT_FEATURE_NUM = 200


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
        part = 5
        for temp in data[0:data_len//part]:
            s1 += cut_word(temp)

        s2 = ""
        for temp in data[data_len//part:data_len*2//part]:
            s2 += cut_word(temp)

        s3 = ""
        for temp in data[data_len*2//part:data_len*3//part]:
            s3 += cut_word(temp)
        
        s4 = ""
        for temp in data[data_len*3//part:]:
            s4 += cut_word(temp)

        lis.append(s1)
        lis.append(s2)
        lis.append(s3)
        lis.append(s4)
        transfer = TfidfVectorizer(max_features=TEXT_FEATURE_NUM)
        try:
            X = transfer.fit_transform(lis).toarray()
        except:
            X = None
    return X 
