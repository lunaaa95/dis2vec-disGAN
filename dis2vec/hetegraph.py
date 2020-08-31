from networkx.classes.graph import Graph
from networkx.exception import NetworkXError
import networkx as nx
import random
from itertools import cycle,dropwhile
from functools import reduce
import operator
#import logging

#logger = logging.getLogger(__name__)


def gen_filter_func(key, op, val):
    def get_op(inp, relate, cut):
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq,
               '!=': operator.ne}
        return ops[relate](inp, cut)
    return lambda x: get_op(x[key],op,val)

def combine_func(funcs):
    return reduce(lambda fx,fy: lambda x: fx(x) and fy(x), funcs)


class HeteGraph(Graph):
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
            neighbors = list(self.adj[n])
            if filterFunc:
                neighbors = list(filter(lambda x:filterFunc(self.node[x]), neighbors))

            neighbors_dict = [(x,self.node[x]) for x in neighbors]
            if data is True:
                rst = neighbors_dict
            elif data is not False:
                rst = [ (x, ddict[data]) if data in ddict else (x,default) for (x,ddict) in neighbors_dict]
            else:
                rst = neighbors

            return rst
        except KeyError:
            raise NetworkXError("The node %s is not in the graph." % (n,))


    def load_edgelist(self, file_):
        with open(file_) as f:
            cnt = 0
            for l in f:
                node1, node1_type,  \
                node2, node2_type,  weight \
                 = l.strip().split(',')
                self.add_node(node1, type=node1_type)
                self.add_node(node2, type=node2_type)
                self.add_edge(node1, node2, weight=float(weight))
                cnt += 1

    def load_dataframe(self, df):
        def load_single_entry(x):
            self.add_node(x['node1'], type=x['node1_type'])
            self.add_node(x['node2'], type=x['node2_type'])
            self.add_edge(x['node1'], x['node2'], weight=float(x['weight']))
        df.apply(load_single_entry, axis=1)

    

    def random_walk(self, metapath, path_length, start, rand=random.Random()):
        G = self

        if metapath:
            path_type_iter = dropwhile(lambda x: x!=G.node[start]['type'],cycle(metapath))
            next(path_type_iter)

        path = [start]

        while len(path) < path_length:
            cur = path[-1]
            filterFunc = None
            if metapath:
                cur_type = G.node[cur]['type']
                next_type = next(path_type_iter)
                filterFunc = gen_filter_func('type', '==', next_type)
            
            neighbors = G.neighbors(cur, filterFunc=filterFunc)

            if len(neighbors)>0:
                #weighted = {k:v['weight'] if k in neighbors else 0 for (k,v) in G.edge[cur].items()} #hetewalk
                weighted = {k: 1 if k in neighbors else 0 for (k, v) in G.edge[cur].items()} #metapath2vec
                next_node = nx.utils.weighted_choice(weighted)
                path.append(next_node)
            else:
                return None
        return path



    def build_corpus(self, num_walks, path_length, metapath=None, rand=random.Random()):
        G = self
        nodes = G.nodes()
        walks = []
        for cnt in range(num_walks):
            #logger.info("Walk Iter:{cnt}/{total}".format(cnt=cnt+1,total=num_walks))
            rand.shuffle(nodes)
            for node in nodes:
                if metapath and G.node[node]['type'] not in metapath:
                    continue
                walk = G.random_walk(metapath, path_length, node)
                if walk:
                    walks.append(walk)
        return walks


############ end of Class HeteGraph ################







