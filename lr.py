import os
import glob
from sklearn.base import _pprint
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
import numpy as np


def create_dataset():
    path=r"C:\Users\Parsa\Desktop\University\Data Mining\Project\Classification_treev3\Classification_tree\op_spam_v1.4\op_spam_v1.4\negative_polarity"
    n=1
    lst_comment=[]
    lst_class=[]
    lst_type=[]
    # find deceptive comment in dataset and give type(test or train) to it 
    while n<6:
        for child in os.scandir(path):
            if 'deceptive' in child.path:
                    for file in os.listdir(child.path+r'\fold{}'.format(n)):
                        if file.endswith(".txt"):
                            with open(child.path+r'\fold{}'.format(n)+"\\"+file) as f:
                                lines=f.readlines()
                                lst_comment.append(lines)
                                lst_class.append('deceptive')
        lst_type.clear()                        
        if n==5:
            for i in range(len(lst_class)):
                lst_type.append("test")
            dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
        else:
            for i in range(len(lst_class)):
                lst_type.append("train")
            dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
            
        n=n+1
    n=1
    
    # find truthful comment in dataset and give type(test or train) to it
    while n<6:
        for child in os.scandir(path):
            if 'truthful' in child.path:
                    for file in os.listdir(child.path+r'\fold{}'.format(n)):
                        if file.endswith(".txt"):
                            with open(child.path+r'\fold{}'.format(n)+"\\"+file) as f:
                                lines=f.readlines()
                                lst_comment.append(lines)
                                lst_class.append('truthful')   
        lst_type.clear()
        if n==5:
            for i in range(0,len(lst_class)):
                lst_type.append("test")
            dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
        else:
            for i in range(0,len(lst_class)):
                lst_type.append("train")
            dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
        n=n+1
    return dt

def encode_label(dataset):
    le= preprocessing.LabelEncoder()
    le.fit(dataset)
    #encode label
    labels=le.transform(dataset)
    return labels    
def vectorize(dt):
    #vectorize feactures and labels to change the text to the number
    vectorizer = CountVectorizer(min_df=5, encoding='latin-1', ngram_range=(1, 5), stop_words='english')
    features = pd.DataFrame(vectorizer.fit_transform(dt['comment'].values).toarray().astype(np.float32)).values
    labels=encode_label(dt['class'])
     #prepare dataset
    dt['fearures']=features
    return dt
dt= create_dataset()
dt=vectorize(dt)
# x_train=dt_train['comment']
# y_train=dt_train['class']
# lr=LogisticRegression(penalty="l2")
# lr.fit(x_train,y_train)
# print(lr.score())


