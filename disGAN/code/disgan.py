#主文件
import os
import sys
import tensorflow as tf
import pandas as pd
import config
import generator
import discriminator
from utils import ConsGraph
import time
import numpy as np
from link_evaluation import Link_evaluation


class Model():
    def __init__(self, rate, train_gd,dim):

        t = time.time()
        # print "reading graph..."
        # def read_graph(self,filename,rel,lost_type=False,train_rate=1)
        # self.n_node, len(self.relations), self.graph, self.weights

        # d-d:0
        # m-m:1
        # g-g:2
        # m-g:3
        # g-m:4
        # m-d:5
        # d-m:6
        # g-d:7
        # d-g:8
        c = ConsGraph()
        c.read_data('../data/DD_MinMiner.csv', rel=[0, 0])
        c.read_data('../data/MM_misim.csv', rel=[1, 1], lost_type=True)
        c.read_data('../data/GG_HPRD.csv', rel=[2, 2])
        c.read_data('../data/MG_miRTarBase_strong.csv', rel=[3, 4])  # MG_miRTarBase_strong&weak
        c.read_data('../data/MG_miRTarBase_weak.csv', rel=[3, 4])  # MG_miRTarBase_strong&weak
        if train_gd:
            c.read_data('../data/MD_Verified242_miRNet.csv', rel=[5, 6])  # MD_Verified242_miRNet
            c.read_data('../data/GD_DisGeNET.csv', rel=[7, 8], train_rate=rate)  # GD_DisGeNET
            pretrain_node_emb_filename = "../pretrain/emb_gd" + str(rate)+str(dim)+".pkl"
        else:
            c.read_data('../data/MD_Verified242_miRNet.csv', rel=[5, 6], train_rate=rate)  # MD_Verified242_miRNet
            c.read_data('../data/GD_DisGeNET.csv', rel=[7, 8])  # GD_DisGeNET
            pretrain_node_emb_filename = "../pretrain/emb_md" + str(rate)+str(dim)+".pkl"

        c.cons_graph()

        #self.datatype=c.data_type()
        # c.read_graph(config.graph_filename)
        self.n_node, self.n_relation, self.graph, self.weights = c.n_node, c.n_relation, c.graph, c.weights
        self.dic = c.dic
        # self.n_relation=len(self.relations)
        self.node_list = list(self.graph.keys())  # range(0, self.n_node)
        print('[%.2f] reading graph finished. #node = %d #relation = %d #edges = %d' % (
        time.time() - t, self.n_node, self.n_relation, len(self.weights)))
        self.rate = rate
        t = time.time()
        # print "read initial embeddings..."
        self.node_embed_init_d = c.read_embeddings(filename=pretrain_node_emb_filename,
                                                   n_embed=dim)
        self.node_embed_init_g = c.read_embeddings(filename=pretrain_node_emb_filename,
                                                   n_embed=dim)

        # self.rel_embed_init_d = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_d,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        # self.rel_embed_init_g = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_g,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        print("[%.2f] read initial embeddings finished." % (time.time() - t))

        print("build GAN model...")
        self.dim=dim
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        self.saver = tf.train.Saver()

        self.link_prediction = Link_evaluation(c.test, c.dic, c.df, train_gd)#--------association prediction

        #    self.dblp_evaluation = DBLP_evaluation()
        # self.yelp_evaluation = Yelp_evaluation()
        #   self.aminer_evaluation = Aminer_evaluation()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

        # self.show_config()

        # def show_config(self):
        #      print '--------------------'
        #      print 'Model config : '
        #      print 'dataset = ', config.dataset
        #      print 'batch_size = ', config.batch_size
        #      print 'lambda_gen = ', config.lambda_gen
        #      print 'lambda_dis = ', config.lambda_dis
        #

    # print 'n_sample = ', config.n_sample
    #      print 'lr_gen = ', config.lr_gen
    #      print 'lr_dis = ', config.lr_dis
    #      print 'n_epoch = ', config.n_epoch
    #      print 'd_epoch = ', config.d_epoch
    #      print 'g_epoch = ', config.g_epoch
    #      print 'n_emb = ', config.n_emb
    #      print 'sig = ', config.sig
    #      print 'label smooth = ', config.label_smooth
    #      print '--------------------'

    def build_generator(self):
        # with tf.variable_scope("generator"):
        self.generator = generator.Generator(n_node=self.n_node,
                                             n_relation=self.n_relation,
                                             rate=self.rate,
                                             node_emd_init=self.node_embed_init_g,
                                             relation_emd_init=None,dim=self.dim)

    def build_discriminator(self):
        # with tf.variable_scope("discriminator"):
        self.discriminator = discriminator.Discriminator(n_node=self.n_node,
                                                         n_relation=self.n_relation,
                                                         rate=self.rate,
                                                         node_emd_init=self.node_embed_init_d,
                                                         relation_emd_init=None,
                                                         dim=self.dim)

    def train(self,train_gd,rate):

        print('start traning...')
        for epoch in range(config.n_epoch):
            print('epoch %d' % epoch)
            t = time.time()

            one_epoch_gen_loss = 0.0
            one_epoch_dis_loss = 0.0
            one_epoch_batch_num = 0.0

            # D-step
            # t1 = time.time()
            for d_epoch in range(config.d_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_dis_loss = 0.0
                one_epoch_pos_loss = 0.0
                one_epoch_neg_loss_1 = 0.0
                one_epoch_neg_loss_2 = 0.0

                for index in range(int(len(self.node_list) / config.batch_size) + 1):
                    # t1 = time.time()
                    pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, neg_node_ids_3, neg_relation_ids_3, node_fake_neighbor_embedding, node_fake_norel_embedding = self.prepare_data_for_d(
                        index)
                    # t2 = time.time()
                    # print t2 - t1
                    _, dis_loss, pos_loss, neg_loss_1, neg_loss_2 = self.sess.run(
                        [self.discriminator.d_updates, self.discriminator.loss, self.discriminator.pos_loss,
                         self.discriminator.neg_loss_1, self.discriminator.neg_loss_2],
                        feed_dict={self.discriminator.pos_node_id: np.array(pos_node_ids),
                                   self.discriminator.pos_relation_id: np.array(pos_relation_ids),
                                   # self.discriminator.pos_weight: np.array(pos_weights),
                                   self.discriminator.pos_node_neighbor_id: np.array(pos_node_neighbor_ids),
                                   self.discriminator.neg_node_id_1: np.array(neg_node_ids_1),
                                   self.discriminator.neg_relation_id_1: np.array(neg_relation_ids_1),
                                   self.discriminator.neg_node_neighbor_id_1: np.array(neg_node_neighbor_ids_1),
                                   self.discriminator.neg_node_id_2: np.array(neg_node_ids_2),
                                   self.discriminator.neg_relation_id_2: np.array(neg_relation_ids_2),
                                   self.discriminator.neg_node_id_3: np.array(neg_relation_ids_3),
                                   self.discriminator.neg_relation_id_3: np.array(neg_relation_ids_3),
                                   self.discriminator.node_fake_neighbor_embedding: np.array(
                                       node_fake_neighbor_embedding),
                                   self.discriminator.node_fake_norel_embedding: np.array(node_fake_norel_embedding)
                                   })

                    one_epoch_dis_loss += dis_loss
                    one_epoch_pos_loss += pos_loss
                    one_epoch_neg_loss_1 += neg_loss_1
                    one_epoch_neg_loss_2 += neg_loss_2

            # G-step

            for g_epoch in range(config.g_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_gen_loss = 0.0

                for index in range(int(len(self.node_list) / config.batch_size)):
                    gen_node_ids, gen_relation_ids, gen_noise_embedding, gen_dis_node_embedding, gen_dis_relation_embedding = self.prepare_data_for_g(
                        index)
                    t2 = time.time()

                    _, gen_loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                                feed_dict={self.generator.node_id: np.array(gen_node_ids),
                                                           self.generator.relation_id: np.array(gen_relation_ids),
                                                           # self.generator.pos_weight: np.array(pos_weights),
                                                           self.generator.noise_embedding: np.array(
                                                               gen_noise_embedding),
                                                           self.generator.dis_node_embedding: np.array(
                                                               gen_dis_node_embedding),
                                                           self.generator.dis_relation_embedding: np.array(
                                                               gen_dis_relation_embedding)})

                    one_epoch_gen_loss += gen_loss

            one_epoch_batch_num = len(self.node_list) / config.batch_size

            # print t2 - t1
            # exit()
            print('[%.2f] gen loss = %.4f, dis loss = %.4f pos loss = %.4f neg loss-1 = %.4f neg loss-2 = %.4f' % \
                  (time.time() - t, one_epoch_gen_loss / one_epoch_batch_num, one_epoch_dis_loss / one_epoch_batch_num,
                   one_epoch_pos_loss / one_epoch_batch_num, one_epoch_neg_loss_1 / one_epoch_batch_num,
                   one_epoch_neg_loss_2 / one_epoch_batch_num))

            # if rate==0.7:
            #     emb_dfg, emb_dfd,rel_dfg,rel_dfd=self.get_embed()
            #     if train_gd:
            #         s="gd"
            #     else:
            #         s="md"
            #     emb_dfg.to_pickle("../res/emb_g" + s + str(epoch)  +".pkl")
            #     emb_dfd.to_pickle("../res/emb_d" + s + str(epoch) + ".pkl")
            #     rel_dfg.to_pickle("../res/remb_g" + s + str(epoch)  +".pkl")
            #     rel_dfd.to_pickle("../res/remb_d" + s + str(epoch) + ".pkl")
            genauc, disauc = self.evaluate_link_prediction() #-----------Auc results
            # wgenauc,wdisauc=self.evaluate_weight_prediction()
            print('gen auc=%.4f,dis auc=%.4f' % (genauc, disauc))
            # print('dis-norel-cos=%.4f,dis-rel-dot=%.4f, dis_rel-cosdis=%.4f' %(disauc[0], disauc[1], disauc[2]))
            #print('wgen-norel-cos=%.4f,wgen-rel-dot=%.4f, wgen_rel-cosdis=%.4f' % (wgenauc[0], wgenauc[1], wgenauc[2]))
            #print('wdis-norel-cos=%.4f,wdis-rel-dot=%.4f, wdis_rel-cosdis=%.4f' % (wdisauc[0], wdisauc[1], wdisauc[2]))
            # if config.dataset == 'dblp':
            #     gen_nmi, dis_nmi = self.evaluate_author_cluster()
            #     print ('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
            #     micro_f1s, macro_f1s = self.evaluate_author_classification()
            #     print ('Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1]))
            #     print ('Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1]))
            # elif config.dataset == 'yelp':
            #     gen_nmi, dis_nmi = self.evaluate_business_cluster()
            #     print ('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
            #     micro_f1s, macro_f1s = self.evaluate_business_classification()
            #     print ('Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1]))
            #     print ('Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1]))
            # elif config.dataset == 'aminer':
            #     gen_nmi, dis_nmi = self.evaluate_paper_cluster()
            # print 'Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi)
            # micro_f1s, macro_f1s = self.evaluate_paper_classification()
            # print 'Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1])
            # print 'Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1])

            # self.evaluate_aminer_link_prediction()
            # self.write_embeddings_to_file(epoch)
            # os.system('python ../evaluation/lp_evaluation_2.py')

        print("training completes")

    def prepare_data_for_d(self, index):

        pos_node_ids = []
        pos_relation_ids = []
        pos_node_neighbor_ids = []
        # pos_weights=[]
        # real node and wrong relation
        neg_node_ids_1 = []
        neg_relation_ids_1 = []
        neg_node_neighbor_ids_1 = []

        # fake node and true relation
        neg_node_ids_2 = []
        neg_relation_ids_2 = []
        node_fake_neighbor_embedding = None

        neg_node_ids_3 = []
        neg_relation_ids_3 = []
        end = (index + 1) * config.batch_size
        if (index + 1) * config.batch_size >= len(self.node_list):
            end = len(self.node_list)
        for node_id in self.node_list[index * config.batch_size: end]:
            for i in range(config.n_sample):

                # sample real node and true relation
                relations = list(self.graph[node_id].keys())
                if len(relations) <= 0:
                    break
                relation_id = relations[np.random.randint(0, len(relations))]
                neighbors = list(self.graph[node_id][relation_id])
                if len(neighbors) <= 0:
                    break
                node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                pos_node_ids.append(node_id)
                pos_relation_ids.append(relation_id)
                pos_node_neighbor_ids.append(node_neighbor_id)
                # pos_weights.append(self.weights[node_id][node_neighbor_id])

                # sample real node and wrong relation
                neg_node_ids_1.append(node_id)
                neg_node_neighbor_ids_1.append(node_neighbor_id)
                neg_relation_id_1 = np.random.randint(0, self.n_relation)
                while neg_relation_id_1 == relation_id:
                    neg_relation_id_1 = np.random.randint(0, self.n_relation)
                neg_relation_ids_1.append(neg_relation_id_1)

                # sample fake node and true relation
                neg_node_ids_2.append(node_id)
                neg_relation_ids_2.append(relation_id)

                # sample fake node and fake relation
                neg_node_ids_3.append(node_id)
                neg_relation_id_3 = np.random.randint(0, self.n_relation)
                while neg_relation_id_3 == relation_id:
                    neg_relation_id_3 = np.random.randint(0, self.n_relation)
                neg_relation_ids_3.append(neg_relation_id_3)

        # generate fake node
        noise_embedding = np.random.normal(0.0, config.sig, (len(neg_node_ids_2), self.dim))

        node_fake_neighbor_embedding = self.sess.run(self.generator.node_neighbor_embedding,
                                                     feed_dict={self.generator.node_id: np.array(neg_node_ids_2),
                                                                self.generator.relation_id: np.array(
                                                                    neg_relation_ids_2),
                                                                self.generator.noise_embedding: np.array(
                                                                    noise_embedding)})
        node_fake_norel_embedding = self.sess.run(self.generator.node_norel_embedding,
                                                  feed_dict={self.generator.node_id: np.array(neg_node_ids_3),
                                                             self.generator.relation_id: np.array(neg_relation_ids_3),
                                                             self.generator.noise_embedding: np.array(noise_embedding)})

        return pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, \
               neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, neg_node_ids_3, neg_relation_ids_3, node_fake_neighbor_embedding, node_fake_norel_embedding

    def prepare_data_for_g(self, index):
        node_ids = []
        relation_ids = []
        end = (index + 1) * config.batch_size
        if (index + 1) * config.batch_size >= len(self.node_list):
            end = len(self.node_list)
        # pos_weights=[]
        for node_id in self.node_list[index * config.batch_size: end]:
            for i in range(config.n_sample):
                relations = list(self.graph[node_id].keys())
                if len(relations) <= 0:
                    break
                relation_id = relations[np.random.randint(0, len(relations))]

                # neighbors = list(self.graph[node_id][relation_id])
                # if len(neighbors)<=0:
                #     break
                # node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]
                # pos_weights.append(self.weights[node_id][node_neighbor_id])
                node_ids.append(node_id)
                relation_ids.append(relation_id)
                # pos_weights.append(self.weights[node_id][self.graph[node_id][]])
        noise_embedding = np.random.normal(0.0, config.sig, (len(node_ids), self.dim))

        dis_node_embedding, dis_relation_embedding = self.sess.run(
            [self.discriminator.pos_node_embedding, self.discriminator.pos_relation_embedding],
            feed_dict={self.discriminator.pos_node_id: np.array(node_ids),
                       self.discriminator.pos_relation_id: np.array(relation_ids)})
        return node_ids, relation_ids, noise_embedding, dis_node_embedding, dis_relation_embedding

    def evaluate_link_prediction(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            relation_matrix = self.sess.run(modes[i].relation_embedding_matrix)
            score = self.link_prediction.auc_score(embedding_matrix, relation_matrix)
            scores.append(score)
        return scores

    # def evaluate_weight_prediction(self):
    #     modes = [self.generator, self.discriminator]
    #     scores = []
    #     for i in range(2):
    #         embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
    #         relation_matrix = self.sess.run(modes[i].relation_embedding_matrix)
    #         score1,score2,score3 = self.link_prediction.weight_auc(embedding_matrix,relation_matrix)
    #         scores.append([score1,score2,score3])
    #     return scores


    def evaluate_author_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.dblp_evaluation.evaluate_author_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_author_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.dblp_evaluation.evaluate_author_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_paper_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.aminer_evaluation.evaluate_paper_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_paper_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.aminer_evaluation.evaluate_paper_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_business_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.yelp_evaluation.evaluate_business_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_business_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.yelp_evaluation.evaluate_business_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_yelp_link_prediction(self):
        modes = [self.generator, self.discriminator]

        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)

            # score = self.yelp_evaluation.evaluate_business_cluster(embedding_matrix)
            # print '%d nmi = %.4f' % (i, score)

            auc, f1, acc = self.yelp_evaluation.evaluation_link_prediction(embedding_matrix)

            #    print 'auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc)

    def evaluate_dblp_link_prediction(self):
        modes = [self.generator, self.discriminator]

        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            # relation_matrix = self.sess.run(modes[i].relation_embedding_matrix)

            auc, f1, acc = self.dblp_evaluation.evaluation_link_prediction(embedding_matrix)

            #   print 'auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc)

    def get_embed(self):
    #-----------validation for m-d(first get emd_vec_df)---
        modes = [self.generator, self.discriminator]

        gembedding_matrix=self.sess.run(self.generator.node_embedding_matrix)
        grelation_matrix = self.sess.run(self.generator.relation_embedding_matrix)

        dembedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        drelation_matrix = self.sess.run(self.discriminator.relation_embedding_matrix)
        grel=grelation_matrix
        drel = drelation_matrix
        emb_dfg = pd.DataFrame(columns=['node', 'vector'])
        rel_dfg=pd.DataFrame(columns=['vector'])
        emb_dfd = pd.DataFrame(columns=['node', 'vector'])
        rel_dfd=pd.DataFrame(columns=['vector'])
        for i in range(9):
            grvec=grel[i]
            drvec=drel[i]
            rel_dfg = rel_dfg.append({'vector': grvec}, ignore_index=True)
            rel_dfd = rel_dfd.append({'vector': drvec}, ignore_index=True)
        for id in range(self.n_node):
            node=list(self.dic.keys())[list(self.dic.values()).index(id)]
            gvec=gembedding_matrix[id]
            dvec = dembedding_matrix[id]
            emb_dfg=emb_dfg.append({'node': node, 'vector': gvec}, ignore_index=True)
            emb_dfd = emb_dfd.append({'node': node, 'vector': dvec}, ignore_index=True)
                #type=self.datatype.loc[self.datatype.node1==node,'node1_type']
            #emb_df = pd.merge(emb_df, self.datatype, on=['node'], how='left')
        return emb_dfg,emb_dfd,rel_dfg,rel_dfd


    def write_embeddings_to_file(self, epoch):
        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            #relation_matrix=self.sess.run(modes[i].rela)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in
                             embedding_list]

            with open(config.emb_filenames[i], 'w') as f:
                lines = [str(self.n_node) + ' ' + str(self.dim) + '\n'] + embedding_str
                f.writelines(lines)


if __name__ == '__main__':
    # for rate in [0.5,0.6,0.7,0.8,0.9]:
    #     model = Model(rate)
    #     model.train()
    #-----参数"gd"表明进行基因-疾病关联预测
    if sys.argv[1] == "gd":
        train_gd = True
    #-------miRNA-疾病关联预测
    else:
        train_gd = False
    # res=[]

    for rate in [0.5,0.6,0.7,0.8,0.9]:
        for dim in [64]:
        # for d_epochs in [1, 5, 10, 15, 20]:
        #     g_epochs = 1
            # for g_epochs in [1,5,10,15,20]:
        #print(d_epochs, g_epochs)
            model = Model(rate, train_gd,dim)
            model.train(train_gd,rate)
