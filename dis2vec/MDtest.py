#miRNA疾病关联预测实验
from dis2vec import *
import logging
import matplotlib.pyplot as plt


paths = [('miRNA','miRNA','disease'),
        ('miRNA','disease','disease'),
        ('miRNA','miRNA','disease','disease'),
        ('miRNA','disease','miRNA','disease'),
        ('miRNA','gene','disease'),
        ('miRNA','gene','gene','disease'),
        ('miRNA','miRNA','gene','disease'),
        ('miRNA','gene','disease','disease')]

#[2]0.91426496319,0.949650159209,0.941581453727,0.966899104683,0.971991037132





def train(rate, selected_paths_idx,dim):

    h = Dis2vec()

    # data format "node1,type1,node2,type2,score"
    h.load_edges('./data/DD_MinMiner.csv')
    h.load_edges('./data/MM_misim.csv', lost_type=True)
    h.load_edges('./data/MD_Verified242_miRNet.csv',train_rate=rate)#MD_Verified242_miRNet
    h.load_edges('./data/GG_HPRD.csv')
    h.load_edges('./data/MG_miRTarBase_strong.csv') #MG_miRTarBase_strong&weak
    h.load_edges('./data/MG_miRTarBase_weak.csv')  # MG_miRTarBase_strong&weak
    h.load_edges('./data/GD_DisGeNET.csv') #GD_DisGeNET

    h.init_model(dim)

    if not selected_paths_idx:
        h.train_model(10,10,5,None, num_iter=1, rewalk=True)
    else:
        selected_paths = [paths[i] for i in selected_paths_idx]
        for path in selected_paths:
            h.train_model(10,len(path),5,path, num_iter=1, rewalk=True)

    #h.gen_test()
    #h.eval_data['score'] = h.eval_data.apply(lambda x: h.eval_sim(x.node1, x.node2), axis=1)
    #auc = h.auc_score(h.eval_sim)
    emb_df = h.get_emb_df()
    auc = h.auc_score(h.eval_sim)
    #fpr,tpr = h.roc(h.eval_sim)
    #return auc,emb_df
    return auc,emb_df
    #fpr,tpr = h.roc(h.eval_sim)

aucs=[]
# f1s=[]
accus=[]
for rate in [0.5,0.6,0.7,0.8,0.9]:
    for dim in [64]:
        selected_paths_idx = [2]
        auc,emb_df = train(rate, selected_paths_idx,dim)
        print(auc)
    #auc, f1, accu, emb_df = train(rate, selected_paths_idx)
        #print(auc)
        aucs.append(auc)
    # f1s.append(f1)
    # accus.append(accu)
        #emb_df.to_pickle("./res/emb_md" + str(rate) +str(dim) +".pkl")
#print(aucs)
    #emb_df.to_pickle("./res/emb_md" + str(rate) + ".pkl")
print(aucs)