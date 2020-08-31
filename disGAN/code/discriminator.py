#判别器
import tensorflow as tf
import config

class Discriminator():
    def __init__(self, n_node, n_relation, rate,node_emd_init, relation_emd_init,dim=""):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]


        self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding'+str(rate)+str(dim),
                                                     shape = self.node_emd_init.shape,
                                                     initializer = tf.constant_initializer(self.node_emd_init),
                                                     trainable = True)
        self.relation_embedding_matrix = tf.get_variable(name = 'dis_relation_embedding'+str(rate)+str(dim),
                                                         shape = [self.n_relation, self.emd_dim, self.emd_dim],
                                                         initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                         trainable = True)


        self.pos_node_id = tf.placeholder(tf.int32, shape = [None])
        self.pos_relation_id = tf.placeholder(tf.int32, shape = [None])
        self.pos_node_neighbor_id = tf.placeholder(tf.int32, shape = [None])

        self.neg_node_id_1 = tf.placeholder(tf.int32, shape = [None])
        self.neg_relation_id_1 = tf.placeholder(tf.int32, shape = [None])
        self.neg_node_neighbor_id_1 = tf.placeholder(tf.int32, shape = [None])

        self.neg_node_id_2 = tf.placeholder(tf.int32, shape = [None])
        self.neg_relation_id_2 = tf.placeholder(tf.int32, shape = [None])
        self.node_fake_neighbor_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])

        self.neg_node_id_3=tf.placeholder(tf.int32, shape = [None])
        self.neg_relation_id_3= tf.placeholder(tf.int32, shape = [None])
        self.node_fake_norel_embedding=tf.placeholder(tf.float32, shape = [None, self.emd_dim])


        self.pos_node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_id)
        self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_neighbor_id)
        self.pos_relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.pos_relation_id)

        self.neg_node_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_1)
        self.neg_node_neighbor_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_neighbor_id_1)
        self.neg_relation_embedding_1 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_1)

        self.neg_node_embedding_2 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_2)
        self.neg_relation_embedding_2 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_2)

        self.neg_node_embedding_3 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_3)
        self.neg_relation_embedding_3 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_3)

        #pos loss通过正确关系连接的真实节点
        t = tf.reshape(tf.matmul(tf.expand_dims(self.pos_node_embedding, 1), self.pos_relation_embedding), [-1, self.emd_dim])
        self.pos_score = tf.reduce_sum(tf.multiply(t, self.pos_node_neighbor_embedding), axis = 1)
        # self.pos_loss = tf.reduce_sum(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.convert_to_tensor(self.pos_weight), logits=self.pos_score))
        self.pos_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pos_score), logits=self.pos_score))

        #neg loss_1通过错误关系连接的真实节点
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_1, 1), self.neg_relation_embedding_1), [-1, self.emd_dim])
        self.neg_score_1 = tf.reduce_sum(tf.multiply(t, self.neg_node_neighbor_embedding_1), axis = 1)
        self.neg_loss_1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_1), logits=self.neg_score_1))

        #neg loss_2通过正确关系连接的伪节点
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2), [-1, self.emd_dim])
        self.neg_score_2 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis = 1)
        self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_2), logits=self.neg_score_2))

        #neg loss_3通过错误关系连接的伪节点
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_3, 1), self.neg_relation_embedding_3), [-1, self.emd_dim])
        self.neg_score_3 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis = 1)
        self.neg_loss_3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_3), logits=self.neg_score_3))


        self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2 + self.neg_loss_3

        optimizer = tf.train.AdamOptimizer(config.lr_dis)

        self.d_updates = optimizer.minimize(self.loss)

