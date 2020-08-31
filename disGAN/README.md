## 命令运行

cd code

python disgan.py gd
python disgan.py md


## Evironment Setting
Python 3.6.3
tensorflow-gpu (1.9.0)
numpy (1.13.3)
scikit-learn (0.19.1)


## Parameter Setting (see config.py)
batch_size : The size of batch.

lambda_gen, lambda_dis : The regularization for generator and discriminator, respectively.

lr_gen, lr_dis : The learning rate for generator and discriminator, respectively.

n_epoch : The maximum training epoch.

sig : The variance of gaussian distribution in generator.

g_epoch, d_epoch: The number of generator and discriminator training per epoch.

n_sample : The size of sample

n_emb : The embedding size

## Files in the folder

* data/: （数据文件放外面文件夹了，注意运行时数据文件位置）

* code/

* pre_train/: 预训练节点向量，采用欠拟合的dis2vec训练得到










