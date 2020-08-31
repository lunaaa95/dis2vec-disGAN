import pandas as pd
from gensim.models import Word2Vec
import random, logging, os
import itertools
from numpy import dot, exp
from sklearn.metrics import roc_auc_score,roc_curve,f1_score,accuracy_score
from hetegraph import *

class Dis2vec(object):
    def __init__(self):
        self.df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.path_dict = {}
    def load_edges(self, filepath, lost_type=False,train_rate=1, add_test=True):
        if not lost_type:
            df = pd.read_csv(filepath, header=None,
            names=['node1','node1_type',
            'node2','node2_type','weight'], dtype=str)
        else:
            df = pd.read_csv(filepath, header=None,
                             names=['node1',
                                    'node2', 'weight'], dtype=str)
            df['node1_type']='miRNA'
            df['node2_type']='miRNA'
        self.df=self.df.append(df)
        df_train = df.sample(frac=train_rate,random_state=1)
        self.train = self.train.append(df_train)
        if add_test:
            df_test = df.drop(df_train.index)
            self.test = self.test.append(df_test)


    def build_graph(self):
        G = HeteGraph()
        G.load_dataframe(self.train)
        #logger.info("build graph with %i nodes and %i edges.", len(G.nodes()), len(G.edges()))
        self.graph = G

    def gen_test(self, ratio=1):
        #logger.info("generate positive/negative samples.")
        positive = self.test[['node1','node2']]
        positive = positive[positive.node1.isin(self.model.wv.vocab) & positive.node2.isin(self.model.wv.vocab)]
        positive['label'] = 1

        def f(col_name):
            keys = self.test[col_name].unique()
            if len(keys) > 1:
                return None
            else:
                return keys[0]
        node1_type = f('node1_type')
        node2_type = f('node2_type')

        if ratio is None:
            neg_size = None
        else:
            neg_size = ratio * len(self.test)
        negative = self.sample_neg(node1_type, node2_type, neg_size)
        negative['label'] = 0
        self.eval_data = pd.concat([positive,negative])

#实验中采样和正样本对相同数目相同类型的负样本对
    def sample_neg(self, node1_type=None, node2_type=None, size=None):
        def filter_node(typ):
            filterFunc = gen_filter_func('type', '==', typ)
            return self.graph.nodes(filterFunc=filterFunc)

        node1_list = filter_node(node1_type)
        node2_list = filter_node(node2_type)

        node_pairs = pd.DataFrame(list(itertools.product(node1_list, node2_list)), columns=['node1','node2'])
        positive = pd.concat([self.train, self.test])[['node1','node2']]
        #common = node_pairs.merge(positive,on=['node1','node2'])
        negative = node_pairs.loc[~node_pairs.set_index(list(node_pairs.columns)).index.isin(positive.set_index(list(positive.columns)).index)]
        negative = negative[negative.node1.isin(self.model.wv.vocab) & negative.node2.isin(self.model.wv.vocab)]
        if size is None:
            return negative
        else:
            return negative.sample(size)


    def init_model(self, size):
        if not hasattr(self, 'graph'):
            #logger.info("build graph before train model.")
            self.build_graph()

        # if hasattr(self, 'model'):
        #     logger.info("reset model.")
        model = Word2Vec(size=size, min_count=1, sample=0, sg=1, hs=0, negative=5)
        self.model = model

    def train_model(self, num_walks, path_length, window, metapath, num_iter=1, rewalk=True, graph=None):
        if not hasattr(self, 'model'):
            #logger.warning("Please init model before train model.")
            return

        model = self.model
        if not graph:
            graph = self.graph

        if metapath in self.path_dict and not rewalk:
            walks = self.path_dict[metapath]
        else: 
            G = graph
            #logger.info("build_corpus with metapath: %s", metapath)
            walks = G.build_corpus(num_walks, path_length, metapath)
            walks = [list(map(lambda x : str(x), walk)) for walk in walks]
            self.path_dict[metapath] = walks

        #logger.info("train model.")
        model.window = window
        if num_iter:
            model.iter = num_iter
        update = True if model.wv.vocab else False
        model.build_vocab(walks, update=update)
        model.train(walks,total_examples=model.corpus_count,epochs=model.iter)
        self.model = model

#余弦相似度
    def eval_sim(self, x, y):
        if not hasattr(self, 'model'):
            #logger.warning("no model.")
            return
        model = self.model

        if x not in model.wv.vocab or y not in model.wv.vocab:
            rst = -1
        else:
            rst = model.wv.similarity(x,y)   #Cosine similarity
        return rst

    # def eval_prop(self, x, y):
    #     model = self.model
    #     if x not in model.vocab or y not in model.vocab:
    #         rst = -1
    #     else:
    #         rst = 1.0 / (1.0 + exp(-dot(model[x], model.syn1neg[model.vocab[y].index].T)))
    #     return rst

#实验评价指标
    def auc_score(self, func):
        if not hasattr(self, 'eval_data'):
            self.gen_test()
        if len(self.eval_data) <1:
            #logger.warning("no positive/negative samples.")
            return
        y_true = self.eval_data.label
        y_scores = self.eval_data.apply(lambda x: func(x.node1, x.node2), axis=1)
        def prob2bin(x):
            if x>=0.5:
                return 1
            else:
                return 0
        #y_preds=y_scores.apply(lambda x:prob2bin(x))
        return roc_auc_score(y_true, y_scores)
            # ,f1_score(y_true,y_preds),accuracy_score(y_true, y_preds)

    # def roc(self, func):
    #     if not hasattr(self, 'eval_data'):
    #         self.gen_test()
    #     if len(self.eval_data) <1:
    #         #logger.warning("no positive/negative samples.")
    #         return
    #     y_true = self.eval_data.label
    #     y_scores = self.eval_data.apply(lambda x: func(x.node1, x.node2), axis=1)
    #     fpr, tpr, _ = roc_curve(y_true, y_scores)
    #     return fpr, tpr

    def get_emb_df(self):
        model = self.model
        df = pd.DataFrame(columns=['node', 'vector'])
        for v in model.wv.vocab:
            df = df.append({'node': v, 'vector': model.wv[v]}, ignore_index=True)
        return df


    # def get_emb_df_pca(self, n_components=2):
    #     df = self.get_emb_df()
    #     from sklearn.decomposition import PCA
    #     pca = pd.DataFrame(PCA(n_components=n_components).fit_transform(df), index=df.index, columns=['x','y'])
    #     pca['type'] = pca.index.map(lambda x : self.graph.node[x]['type'])
    #     return pca
  

