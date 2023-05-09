# 提取音乐高潮部分

import os
import sys
import csv
from pychorus import find_and_output_chorus

static_feature_file_path = ""

# 加载特征
annotation_data = []
try:
    with open(static_feature_file_path,mode="r") as rf:
        csv_reader = csv.reader(rf)
        for item in csv_reader:
            print(item)
            if len(item)==4 and item[3]:
                annotation_data.append({
                    "name":item[2],
                    "annotation":int(item[3])
                })
except:
    print("标签获取失败")
    exit(0)

def extract_all_file():
    count = 1000
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    for filepath in os.listdir(modpath):
        if not os.path.exists(modpath+"/output"):
            os.makedirs(modpath+"/output")
        if 'wav' in filepath:
            lyric_name = filepath.split(".")[0]
            print("extracting:{}".format(filepath))
            annotation = -1
            for item in annotation_data:
                if item['name'] == (lyric_name+".wav"):
                    annotation = item['annotation']
            if annotation == -1:
                continue
            count += 1
            try:
                find_and_output_chorus(modpath+"/"+filepath,"{}/output/{}.wav".format(modpath,count),30)
            except:
                continue
            print("rewrite lyric:{}.lrc".format(lyric_name))
            lyric_path = "{}/output/{}-{}.txt".format(modpath,count,annotation)
            lyric_file = open(lyric_path,mode="wb")
            with open("{}/{}.lrc".format(modpath,lyric_name),mode="rb") as rf:
                line = rf.readline()
                while line:
                    lyric_file.write(line)
                    line = rf.readline()
            lyric_file.close()


extract_all_file()


