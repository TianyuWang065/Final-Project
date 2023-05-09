import csv ,os

data_path = ""
csv_header = ["text","label"]
annotation_file = ""
arf = open(annotation_file,mode="r",encoding="utf-8")
annotation_reader = list(csv.DictReader(arf))

annotations = []
# 添加标注
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
    annotations.append({"label":label,"music_id":music_id})

lyric_data = []
for annotation in annotations:
    lyric_path = "".format(annotation['music_id'])
    if os.path.exists(lyric_path):
        with open(lyric_path,mode="r",encoding="utf-8") as lyric:
            content = lyric.readline()
            data = {}
            count = 0
            lyric_str = ""
            while content:
                count+=1
                lyric_str += content 
                content = lyric.readline()
                if count % 4 ==3:
                    data["text"] = lyric_str
                    data["label"] = annotation["label"]
                    lyric_data.append(data)
                    lyric_str = ""

with open(data_path, mode="w", encoding="utf-8-sig", newline="") as wf:
    writer = csv.DictWriter(wf, csv_header)
    writer.writeheader()
    writer.writerows(lyric_data)