#读数据建图和预训练的向量
import numpy as np
import pandas as pd


class ConsGraph():
    def __init__(self):
        self.df=pd.DataFrame()
        self.train=pd.DataFrame()
        self.test=pd.DataFrame()
        self.dic={}
        self.relations = set()
        self.nodes = set()
        self.graph = {}
        self.weights={}
        #self.disease={} #for m-d predict validation
    def read_data(self,filename,rel,lost_type=False,train_rate=1):
        # d-d:0
        # m-m:1
        # g-g:2
        # m-g:3
        # g-m:4
        # m-d:5
        # d-m:6
        # g-d:7
        # d-g:8
        #graph_filename = '../data/dblp/dblp_triple.dat'
        #dic:name->id
        #weight[id][id]=weight
        #graph[id][rel]=[id1,id2,.]
        if not lost_type:
            df = pd.read_csv(filename, header=None,names=['node1','node1_type',
            'node2','node2_type','weight'], dtype=str)
        else:
            df=pd.read_csv(filename,header=None,names=['node1','node2','weight'])
            df['node1_type']='miRNA'
            df['node2_type']='miRNA'
        df['relation1']=rel[0]
        df['relation2'] = rel[1]
        self.df=self.df.append(df) #3299278
        if train_rate!=1:
            df_train = df.sample(frac=train_rate, random_state=1)
            self.train = self.train.append(df_train)
            df_test = df.drop(df_train.index)
            self.test = self.test.append(df_test)
        else:
            self.train=self.train.append(df)

    # dataframe with node and its type----------time too long
    # def data_type(self):
    #     self.datatype=self.df[['node1','node1_type']]
    #     for _,row in self.df.iterrows():
    #         self.datatype.append({'node1':row['node2'],'node1_type':row['node2_type']},ignore_index=True)
    #     self.datatype.drop_duplicates(subset=None, keep='first', inplace=True)
    #     self.datatype.rename(columns=lambda x: x.replace('1', ''), inplace=True)
    #     return self.datatype


    def cons_graph(self):
        i=0
        for _,row in self.train.iterrows():
            source_node,target_node,weight,rel1,rel2=row['node1'],row['node2'],row['weight'],row['relation1'],row['relation2']
            if source_node not in self.dic:
                self.dic[source_node]=i
                i+=1
            if target_node not in self.dic:
                self.dic[target_node] = i
                i += 1
            source_id=self.dic[source_node]
            target_id=self.dic[target_node]
            self.nodes.add(source_id)
            self.nodes.add(target_id)

            if source_id not in self.graph:
                self.graph[source_id]={}
            if target_id not in self.graph:
                self.graph[target_id]={}
            if source_id not in self.weights:
                self.weights[source_id]={}
            self.weights[source_id][target_id]=weight
            if target_id not in self.weights:
                self.weights[target_id]={}
            #self.weights[source_id][target_id]=weight
            self.weights[target_id][source_id]=weight

            self.relations.add(rel1)
            self.relations.add(rel2)
            if rel1 not in self.graph[source_id]:
                self.graph[source_id][rel1]=[]
            self.graph[source_id][rel1].append(target_id)
            if rel2 not in self.graph[target_id]:
                self.graph[target_id][rel2]=[]
            self.graph[target_id][rel2].append(source_id)

                #self.graph[target_id][r].append(source_id)
            self.n_node = len(self.nodes)
            self.n_relation=len(self.relations)
                # print relations
        # with open(graph_filename) as infile:
        #     for line in infile.readlines():
        #         source_node, target_node, relation = line.strip().split(' ')
        #         source_node = int(source_node)
        #         target_node = int(target_node)
        #         relation = int(relation)
        #
        #         nodes.add(source_node)
        #         nodes.add(target_node)
        #         relations.add(relation)
        #
        #         if source_node not in graph:
        #             graph[source_node] = {}
        #
        #         if relation not in graph[source_node]:
        #             graph[source_node][relation] = []
        #
        #         graph[source_node][relation].append(target_node)

    def read_embeddings(self, filename,  n_embed):
        df = pd.read_pickle(filename)
        self.embedding_matrix = np.random.rand(len(self.dic), n_embed)
        for index, row in df.iterrows():
            id=self.dic[row['node']]
            emd = row['vector']
            # print(emd)
            self.embedding_matrix[id, :] = emd
        return self.embedding_matrix

# def str_list_to_float(str_list):
#     return [float(item) for item in str_list]
#
# def read_embeddings(filename, n_node, n_embed):
#
#     embedding_matrix = np.random.rand(n_node, n_embed)
#     i = -1
#     with open(filename) as infile:
#         for line in infile.readlines()[1:]:
#             i += 1
#             emd = line.strip().split()
#             embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
#     return embedding_matrix

# if __name__ == '__main__':
#     n_node, n_relation, graph  = read_dblp_graph()

    #embedding_matrix = read_embeddings('../data/dblp/rel_embeddings.txt', 6, 64)
  #  print graph[1][1]
