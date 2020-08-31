#miRNA疾病关联预测实验
from dis2vec import *
import logging
import matplotlib.pyplot as plt
#logging.basicConfig(filename='GDtest.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logger = logging.getLogger(__name__)



paths = [('gene','gene','disease'),
        ('gene','disease','disease'),
        ('gene','gene','disease','disease'),
        ('gene','disease','gene','disease'),
        ('gene','miRNA','disease'),
        ('gene','gene','miRNA','disease'),
        ('gene','miRNA','miRNA','disease'),
        ('gene','miRNA','disease','disease')]



def train(rate, selected_paths_idx,dim):

    h = Dis2vec()

    # data format "node1,type1,node2,type2,score"
    h.load_edges('./data/DD_MinMiner.csv')
    h.load_edges('./data/MM_misim.csv', lost_type=True)
    h.load_edges('./data/MD_Verified242_miRNet.csv')#MD_Verified242_miRNet
    h.load_edges('./data/GG_HPRD.csv')
    h.load_edges('./data/MG_miRTarBase_strong.csv') #MG_miRTarBase_strong&weak
    h.load_edges('./data/MG_miRTarBase_weak.csv')  # MG_miRTarBase_strong&weak
    h.load_edges('./data/GD_DisGeNET.csv',train_rate=rate) #GD_DisGeNET

    h.init_model(dim)

    if not selected_paths_idx:
        #h.train_model(10,10,5,None, num_iter=1, rewalk=True)
        h.train_model(10,10,5,None, num_iter=1, rewalk=True)
    else:
        selected_paths = [paths[i] for i in selected_paths_idx]
        for path in selected_paths:
            h.train_model(10,len(path),5,path, num_iter=1, rewalk=True)

    h.gen_test()
    h.eval_data['score'] = h.eval_data.apply(lambda x: h.eval_sim(x.node1, x.node2), axis=1)
    #logger.info(h.eval_data.groupby('label').apply(lambda x: x.score.mean()))
    #logger.info(h.eval_data.apply(lambda x: x.node1 in h.model.vocab and x.node2 in h.model.vocab, axis=1).value_counts())
    emb_df = h.get_emb_df()
    auc = h.auc_score(h.eval_sim)
    #fpr,tpr = h.roc(h.eval_sim)
    return auc, emb_df

if __name__ == '__main__':
    #rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rates=[0.5,0.6,0.7,0.8,0.9]
    #selected_paths_idxs = [[0],[0,3],[0,3,4,5,6]]
    selected_paths_idxs = [[0]]
    aucs=[]
    # f1s=[]
    # accus=[]
    for selected_paths_idx in selected_paths_idxs:
        for rate in rates:
            for dim in [64]:
                auc,emb_df = train(rate, selected_paths_idx,dim)
                print(auc)
                aucs.append(auc)
            # f1s.append(f1)
            # accus.append(accu)
                #emb_df.to_pickle("./res/emb_gd"+str(rate)+str(dim)+".pkl")
            # logger.info("selected path: " + str(selected_paths_idx))
            # logger.info("rate: " + str(rate))
            # logger.info("auc: " + str(auc))
            #plt.plot(fpr, tpr)
    print(aucs)
    #plt.show()
# scores=[]
# selected_paths_idx = [0]
# for rate in [0.5,0.6,0.7,0.8,0.9]:
#     auc = train(rate, selected_paths_idx)
#     print(auc)
#     scores.append((rate,auc))
# scores
#[0,3],0.9 0.71043680667


#[2],0.9 0.702614279204
#[5] 0.9 0.514724133419
#[1] 0.9 0.593102489223;0.8 0.604011746924
#[0] 0.630701142459,0.668679254017,0.711937957792 ,
#[0][(0.5, 0.63070114245913023), (0.6, 0.66867925401698836), (0.7, 0.71193795779189029), (0.8, 0.71309487756861101), (0.9, 0.74884592016335638)]

#[0]0.645162271251,    0.728728057204,0.753656305335