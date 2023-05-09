class Utils:

    labels = []
    

    def __init__(self) -> None:
        annotation = ""
        arf = open(annotation,mode="r",encoding="utf-8")
        annotation_reader = list(csv.DictReader(arf))

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
                
            self.labels.append({'label':label,'music_id':music_id})

        arf.close()
    

    @staticmethod
    def random_int(start,end):
        return random.randint(start,end) 

    """
    返回CSV文件的表头和
    字典对象列表
    """
    @staticmethod
    def get_csv_dic_list(path):
        file = open(path,mode="r",encoding="utf-8")
        csv_dic = csv.DictReader(file)
        fieldnames = csv_dic.fieldnames
        data = []
        for dic in csv_dic:
            data.append(dict(dic))
        file.close()
        return fieldnames,data 

    def get_label(self,music_id):
        for label in self.labels:
            if label['music_id'] == music_id:
                return label['label']
            

    def multiclass_roc_auc_score(self,y_true, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_true)
        y_true = lb.transform(y_true)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_true, y_pred, average=average)
    
    def save_data_label(self,data,label,filename):
        with open(filename+"data.csv",mode="w",encoding="utf-8") as wf:
            csv_writer = csv.writer(wf)
            for item in data:
                csv_writer.writerow(item)
        
        with open(filename+"label.csv",mode="w",encoding="utf-8") as wf:
            csv_writer = csv.writer(wf)
            csv_writer.writerow(label)
