#关联预测实验
import numpy as np
import pandas as pd
import itertools
import config
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math

class Link_evaluation():
    def __init__(self,test,dic,df,train_gd):

        # def fun(x,y):
        #     return dic[x],dic[y]

        #self.positives=pd.DataFrame(columns=['node1', 'node2', 'label'])
        self.positives = test[['node1','node2','weight']]

        self.positives = self.positives[self.positives.node1.isin(dic) & self.positives.node2.isin(dic)]
        self.pos = pd.DataFrame(columns=['node1', 'node2'])

        self.pos['node1'] = self.positives.apply(lambda x: dic[x['node1']], axis=1)
        self.pos['node2'] = self.positives.apply(lambda x: dic[x['node2']], axis=1)
        self.pos['weight']=self.positives['weight']
        #self.pos = self.pos.apply(lambda x: fun(x['node1'], x['node2']), axis=1)
        #self.negatives=pd.DataFrame(columns=['node1', 'node2', 'label'])
        self.node1_type=test['node1_type'].unique()[0]
        self.node2_type=test['node2_type'].unique()[0]
        # for _,row in test.iterrows():
        #     if row['node1'] in dic and row['node2'] in dic:
        #         source=dic[row['node1']]
        #         target=dic[row['node2']]
        #         self.positives=self.positives.append({'node1':source,'node2':target,'label':1},ignore_index=True)
        node1_list = list(df.loc[df.node1_type == self.node1_type, 'node1'].sample(2 * len(self.positives)))
        node2_list= list(df.loc[df.node2_type == self.node2_type, 'node2'].sample(2 * len(self.positives)))
        node_pairs = pd.DataFrame(list(itertools.product(node1_list, node2_list)), columns=['node1', 'node2'])
        pos=df[['node1','node2']]
        self.negatives=node_pairs.loc[~node_pairs.set_index(list(node_pairs.columns)).index.isin(pos.set_index(list(pos.columns)).index)]
        self.negatives=self.negatives[self.negatives.node1.isin(dic) & self.negatives.node2.isin(dic)]
        self.negatives=self.negatives.sample(len(self.positives))
        self.neg = pd.DataFrame(columns=['node1', 'node2'])
        self.neg['node1'] = self.negatives.apply(lambda x: dic[x['node1']],axis=1)
        self.neg['node2'] = self.negatives.apply(lambda x: dic[x['node2']], axis=1)
        self.neg['weight']=0
        self.pos['label']=1
        self.neg['label']=0
        self.eval_data=pd.concat([self.pos,self.neg])
        self.train_gd=train_gd

    def cosine_sim(self,x, y):
        num = np.dot(x, y)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        cos = 1.0 * num / denom
        # cos = 0.5+0.5*(num / denom)
        return cos
    def cos_dot(self,x,y,relation_matrix):
        if self.train_gd:
            rel=relation_matrix[7]
        else:
            rel=relation_matrix[5]
        res=x.dot(rel).dot(y)
        return res

    def norm_dot(self,x,y,relation_matrix):
        if self.train_gd:
            rel=relation_matrix[7]
        else:
            rel=relation_matrix[5]
        res=x.dot(rel).dot(y)
        denorm=np.linalg.norm(x) * np.linalg.norm(rel)*np.linalg.norm(y)
        return res*1.0/denorm

    def auc_score(self,embedding_matrix,relation_matrix):
        #embedding_list = embedding_matrix.tolist()
        y_true = self.eval_data.label
        #y_scores1 = self.eval_data.apply(lambda x: self.cosine_sim(embedding_matrix[x.node1], embedding_matrix[x.node2]), axis=1)
        #y_scores2=self.eval_data.apply(lambda x: self.cos_dot(embedding_matrix[x.node1], embedding_matrix[x.node2],relation_matrix), axis=1)
        y_scores=self.eval_data.apply(lambda x: self.norm_dot(embedding_matrix[x.node1], embedding_matrix[x.node2],relation_matrix), axis=1)
        return roc_auc_score(y_true, y_scores)

    # def weight_auc(self,embedding_matrix,relation_matrix):
    #     y_true = self.eval_data.weight
    #     y_scores1 = self.eval_data.apply(lambda x: self.cosine_sim(embedding_matrix[x.node1], embedding_matrix[x.node2]), axis=1)
    #     y_scores2=self.eval_data.apply(lambda x: self.cos_dot(embedding_matrix[x.node1], embedding_matrix[x.node2],relation_matrix), axis=1)
    #     y_scores3=self.eval_data.apply(lambda x: self.norm_dot(embedding_matrix[x.node1], embedding_matrix[x.node2],relation_matrix), axis=1)
    #     return roc_auc_score(y_true, y_scores1),roc_auc_score(y_true, y_scores2),roc_auc_score(y_true, y_scores3)


