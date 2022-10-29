import os
from sklearn.base import _pprint
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
import numpy as np


def create_dataset():
    # path=r"C:\Users\Parsa\Desktop\University\Data Mining\Project\Classification_treev3\Classification_tree\op_spam_v1.4\op_spam_v1.4\negative_polarity"
    dirname = os.path.dirname(__file__)
    path= os.path.join(dirname, r"op_spam_v1.4/op_spam_v1.4/negative_polarity")
    n=1
    df = pd.DataFrame(columns=['class','comment','type', 'filename'])
    # find deceptive comment in dataset and give type(test or train) to it 
    while n<6:
        lst_comment=[]
        lst_class=[]
        lst_type=[]
        lst_file=[]
        lst_fold=[]

        for child in os.scandir(path):
            if 'deceptive' in child.path:

                    for file in os.listdir(child.path+r'/fold{}'.format(n)):
                        if file.endswith(".txt"):
                            with open(child.path+r'/fold{}'.format(n)+"/"+file) as f:
                                lines=f.read()
                                lst_comment.append(lines)
                                lst_class.append('deceptive')
                                lst_file.append(file)
                                lst_fold.append(n)
        lst_type.clear()    
        if n==5:
            for i in range(len(lst_class)):
                lst_type.append("test")
            # dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
            df = pd.concat((df, pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type, 'filename':lst_file, 'fold':lst_fold})))
        else:
            
            for i in range(len(lst_class)):
                lst_type.append("train")
            # dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
            df = pd.concat((df, pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type, 'filename':lst_file, 'fold':lst_fold})))

        n=n+1
    n=1
    
    # find truthful comment in dataset and give type(test or train) to it
    while n<6:
        lst_comment=[]
        lst_class=[]
        lst_type=[]
        lst_file=[]
        lst_fold=[]


        for child in os.scandir(path):
            if 'truthful' in child.path:
                    for file in os.listdir(child.path+r'/fold{}'.format(n)):
                        if file.endswith(".txt"):
                            with open(child.path+r'/fold{}'.format(n)+"/"+file) as f:
                                lines=f.read()
                                lst_comment.append(lines)
                                lst_class.append('truthful')
                                lst_file.append(file)
                                lst_fold.append(n)

        lst_type.clear()
        if n==5:
            for i in range(0,len(lst_class)):
                lst_type.append("test")
            # dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
            df = pd.concat((df, pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type, 'filename':lst_file, 'fold':lst_fold})))

        else:
            for i in range(0,len(lst_class)):
                lst_type.append("train")
            # dt=pd.DataFrame(pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type}))
            df = pd.concat((df, pd.DataFrame({'class':lst_class,'comment':lst_comment,'type':lst_type, 'filename':lst_file, 'fold':lst_fold})))


        n=n+1
    return df.reset_index()


if __name__ == "__main__":


    dt = create_dataset()
    dt = dt.drop(['index'], axis=1)

    dt.to_csv('data/original.csv', index=False)

