#基于随机游走的度量选择元路径
import pandas as pd
from networkx.classes.graph import Graph
import networkx as nx
from networkx.exception import NetworkXError
import random
from itertools import cycle,dropwhile
from functools import reduce
import operator
# import logging





ind='./dmg_data.csv' #这个文件是我整合所有数据生成的
df= pd.read_csv(ind, header=None,names=['node1','node1_type','node2','node2_type','weight','edge_type'], dtype=str)

def load_single_entry(x,graph):
    graph.add_node(x['node1'], type=x['node1_type'])
    graph.add_node(x['node2'], type=x['node2_type'])
    graph.add_edge(x['node1'], x['node2'], weight=float(x['weight']))
g=nx.Graph()
df.apply(lambda x:load_single_entry(x,g), axis=1)





def gen_filter_func(key, op, val):
    def get_op(inp, relate, cut):
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq,
               '!=': operator.ne}
        return ops[relate](inp, cut)

    return lambda x: get_op(x[key], op, val)


def combine_func(funcs):
    return reduce(lambda fx, fy: lambda x: fx(x) and fy(x), funcs)


class HeteGraph(Graph):
    # def __init__(self):
    def __init__(self):
        self.dic={}
        self.sum_degree=0
    def build_graph(self,g):
        self.graph = g
        for node in self.graph.nodes():
            self.dic[node]=0
    def nodes_iter(self, data=False, default=None, filterFunc=None):
        if data is True:
            for n, ddict in self.node.items():
                if filterFunc and not filterFunc(self.node[n]):
                    continue
                else:
                    yield (n, ddict)
        elif data is not False:
            for n, ddict in self.node.items():
                if filterFunc and not filterFunc(self.node[n]):
                    continue
                else:
                    d = ddict[data] if data in ddict else default
                    yield (n, d)

        else:
            for n in self.node:
                if filterFunc and not filterFunc(self.node[n]):
                    continue
                else:
                    yield n

    def nodes(self, data=False, default=None, filterFunc=None):
        return list(self.nodes_iter(data=data, default=default, filterFunc=filterFunc))

    def neighbors(self, n, data=False, default=None, filterFunc=None):
        try:
            neighbors = list(self.graph.adj[n])
            if filterFunc:
                neighbors = list(filter(lambda x: filterFunc(self.graph.node[x]), neighbors))

            neighbors_dict = [(x, self.graph.node[x]) for x in neighbors]
            if data is True:
                rst = neighbors_dict
            elif data is not False:
                rst = [(x, ddict[data]) if data in ddict else (x, default) for (x, ddict) in neighbors_dict]
            else:
                rst = neighbors

            return rst
        except KeyError:
            raise NetworkXError("The node %s is not in the graph." % (n,))

    def load_edgelist(self, file_):
        with open(file_) as f:
            cnt = 0
            for l in f:
                node1, node1_type, node2, node2_type, weight = l.strip().split(',')
                self.add_node(node1, type=node1_type)
                self.add_node(node2, type=node2_type)
                self.add_edge(node1, node2, weight=float(weight))
                cnt += 1
                # logger.info("add %i edges.", cnt)

    def load_dataframe(self, df):
        def load_single_entry(x):
            self.graph.add_node(x['node1'], type=x['node1_type'])
            self.graph.add_node(x['node2'], type=x['node2_type'])
            self.graph.add_edge(x['node1'], x['node2'], weight=float(x['weight']))

        df.apply(load_single_entry, axis=1)
        # logger.info("add %i edges.", len(df))

    def random_walk(self, metapath, path_length, start, rand=random.Random()):
        G = self.graph
        dic=self.dic
        cur_deg=0
        if metapath:
            path_type_iter = dropwhile(lambda x: x != G.node[start]['type'], cycle(metapath))
            next(path_type_iter)

        path = [start]
        cur_deg+=G.degree(start)
        self.dic[start] +=1
        while len(path) < path_length:
            cur = path[-1]
            filterFunc = None
            if metapath:
                cur_type = G.node[cur]['type']
                next_type = next(path_type_iter)
                filterFunc = gen_filter_func('type', '==', next_type)

            neighbors = self.neighbors(cur, filterFunc=filterFunc)

            if len(neighbors) > 0:
                weighted = {k: v['weight'] if k in neighbors else 0 for (k, v) in G.edge[cur].items()}
                next_node = nx.utils.weighted_choice(weighted)
                self.dic[next_node]+=1
                cur_deg+=G.degree(next_node)
                path.append(next_node)
            else:
                self.dic=dic
                cur_deg=0
                return None
        cur_deg=cur_deg/path_length
        self.sum_degree+=cur_deg
        return path

    def build_corpus(self, num_walks, path_length, metapath=None, rand=random.Random()):
        G = self.graph
        nodes = G.nodes()
        walks = []
        for cnt in range(num_walks):
            # logger.info("Walk Iter:{cnt}/{total}".format(cnt=cnt+1,total=num_walks))
            rand.shuffle(nodes)
            for node in nodes:
                if metapath and G.node[node]['type'] not in metapath:
                    continue
                walk=self.random_walk(metapath, path_length, node)
                if walk:
                    walks.append(walk)
        return walks





gpaths = [('gene', 'gene', 'disease'),
             ('gene', 'disease', 'disease'),
             ('gene', 'gene', 'disease', 'disease'),
             ('gene', 'disease', 'gene', 'disease'),
             ('gene', 'miRNA', 'disease'),
             ('gene', 'gene', 'miRNA', 'disease'),
             ('gene', 'miRNA', 'miRNA', 'disease'),
             ('gene', 'miRNA', 'disease', 'disease')]
# In [235]: tencount
# Out[235]: [8658, 14422, 9364, 10184, 16103, 10465, 16084, 16460]



paths = [('miRNA', 'miRNA', 'disease'),
             ('miRNA', 'disease', 'disease'),
             ('miRNA', 'miRNA', 'disease', 'disease'),
             ('miRNA', 'disease', 'miRNA', 'disease'),
             ('miRNA', 'gene', 'disease'),
             ('miRNA', 'gene', 'gene', 'disease'),
             ('miRNA', 'miRNA','gene', 'disease'),
             ('miRNA', 'gene','disease', 'disease')]
# In [11]: tencount
# Out[11]: [21323, 19540, 19381, 21335, 14820, 10011, 15481, 14626]

#实验中汇报的是tencount，随机游走10次被访问次数不超过10次的节点数目
degrees=[]
zerocount=[]
tencount=[]
lenwalks=[]
typecount=[]
for path in paths:
    h = HeteGraph()
    h.build_graph(g)
    walks=h.build_corpus(10, len(path), path)
    degrees.append(h.sum_degree)
    lenwalks.append(len(walks))
    zcount=0
    tcount=0
    pcount=0
    for k in h.dic.keys():
        if h.dic[k] ==0:
            zcount += 1
        if h.dic[k]<=10:
            tcount +=1
            if g.node[k]['type']!='gene':
                pcount+=1
    zerocount.append(zcount)
    tencount.append(tcount)
    typecount.append(pcount)


avg_degree=[]
for i in range(len(degrees)):
     t=degrees[i]*1.0/lenwalks[i]
     avg_degree.append(t)

#---------------gene-disease-----------------------
#In [234]: zerocount
#Out[234]: [981, 981, 981, 981, 0, 0, 0, 0]
# In [235]: tencount
# Out[235]: [8658, 14422, 9364, 10184, 16103, 10465, 16084, 16460]
# In [236]: lenwalks
# Out[236]: [81456, 52352, 44178, 114960, 111934, 75297, 94374, 89249]
# In [237]: degrees
# Out[237]: [28712724,60891268,39183913,49592943,97213101,42188517,70326682,105475783]
# In [241]: avg_degree
# Out[241]: [352.49366529169123,1163.1125458435208,886.9553397618724,431.3930323590814,
#            868.4859024067754,560.2947926212199,745.1912814970225,1181.8147318177234]
# In [248]: typecount
# Out[248]: [7612, 13462, 8357, 9198, 16028, 10206, 16018, 16325]



#-------------------miRNA-disease--------------------
# In [10]: zerocount
# Out[10]: [15463, 15463, 15463, 15463, 0, 0, 0, 0]
#
# In [11]: tencount
# Out[11]: [21323, 19540, 19381, 21335, 14820, 10011, 15481, 14626]
#
# In [12]: lenwalks
# Out[12]: [2734, 2952, 2471, 3250, 20309, 18770, 15850, 4923]
#
# In [13]: degrees
# Out[13]: [1935675, 3606017, 2473332, 2934828, 14240435, 10271925, 10523624, 4613551]
# In [15]: avg_degree
# Out[15]:
# [708.001097293343,1221.5504742547425,1000.9437474706597,
# 903.024,701.1883893840169,547.2522642514651,663.9510410094638,937.1421897217144]
# In [28]: typecount
# Out[28]: [5859, 4086, 3906, 5872, 1270, 3278, 1155, 1057]







#('gene','gene','disease') degree 28712380 <=10(10 and 0) 8642; =0 981
#('gene','miRNA','disease') degree 97133664  <=10 le16137   =0 0






