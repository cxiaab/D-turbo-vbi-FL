
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sumproduct import Variable, Factor, FactorGraph

import seaborn as sns
import random
import tempfile
# import tensorflow_model_optimization as tfmot
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import warnings
# matplotlib.use("Qt5Agg")   # �޸����õĺ�� backend ע�� matplotlib.use("Agg")û��

# warnings.filterwarnings('ignore')

# %matplotlib inline
#################CIFAR10#####################################
cifar10 = tf.keras.datasets.cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
train_labels = np.reshape(train_labels, (50000,))
test_labels = np.reshape(test_labels, (10000,))
######### sort training set by the lables 0-9
# idx = np.argsort(train_labels)
# train_features = train_features[idx]
# train_labels = train_labels[idx]
# ################sort test set#############
# idx_test = np.argsort(test_labels)
# test_features = test_features[idx_test]
# test_labels = test_labels[idx_test]
######### shuffle the dataset
idx = np.arange(train_labels.shape[0])  # ֱ�ӵ���ʱ��˳��0��1��2����������60000
np.random.shuffle(idx)  # randomize the idx
print(idx)
train_features = train_features[idx]  # randomized dataset
train_labels = train_labels[idx]  # corresponding training label
#########################################
# train_features = train_features.reshape(60000, 28*28)
# test_features = test_features.reshape(10000, 28*28)
# train_features, test_features = train_features / 255.0, test_features / 255.0  # normalization
###########################################################################################
# train_features = train_features/255.0
test_features = test_features/255.0  # normalization
#####################################################################
# define create_local_dataset
def create_local_dataset(train_features, train_labels, num_clients):
    local_train_features_list = []
    local_train_labels_list = []
    local_dataset_size_list = []
    local_dataset_size = int(len(train_labels) / num_clients)  # local dataset size
    for i in range(num_clients):
        local_train_features_temp = train_features[i*local_dataset_size : (i+1)*local_dataset_size]  # get train features of current client
        local_train_labels_temp = train_labels[i*local_dataset_size : (i+1)*local_dataset_size]  # get train labels of current client
        ####shuffle the current local dataset##########################
        idx = np.arange(local_train_labels_temp.shape[0])
        np.random.shuffle(idx)  # randomize the idx
        local_train_features_temp = local_train_features_temp[idx]  # randomized dataset
        local_train_labels_temp = local_train_labels_temp[idx]  # corresponding training label
        ######local dataset normalization##################
        local_train_features_temp = local_train_features_temp/255.0
        ###################################################
        local_train_features_list.append(local_train_features_temp)
        local_train_labels_list.append(local_train_labels_temp)
        local_dataset_size_list.append(len(local_train_labels_temp))
        assert len(local_train_labels_temp) == int(len(train_labels) / num_clients)

    return local_train_features_list, local_train_labels_list, local_dataset_size_list

def create_local_dataset_noniid(train_features, train_labels, num_clients, shard_num_per_client):
    ################sort the train dataset from 0 to 9###################
    idx = np.argsort(train_labels)
    train_features = train_features[idx]
    train_labels = train_labels[idx]
    #####################################################################
    shard_num = shard_num_per_client*num_clients  # divide the whole dataset into shards, each client has two kinds of labels
    shard_size = int(len(train_labels) / shard_num)  # shard_size
    print('shard_size = ', len(train_labels) / shard_num)
    # local_dataset_size = shard_num_per_client*shard_size
    shard_idx = np.arange(shard_num)
    np.random.shuffle(shard_idx)  # randomize the shard_idx
    local_train_features_list = []
    local_train_labels_list = []
    local_dataset_size = []
    for i in range(num_clients):
        for j in range(shard_num_per_client):
            local_train_features_temp_1 = train_features[shard_idx[shard_num_per_client*i + j]*shard_size : (shard_idx[shard_num_per_client*i+j]+1)*shard_size]  # get train features of current client
            local_train_labels_temp_1 = train_labels[shard_idx[shard_num_per_client*i + j]*shard_size : (shard_idx[shard_num_per_client*i + j]+1)*shard_size]  # get train labels of current client
            if j == 0:
                local_train_features_temp = local_train_features_temp_1
                local_train_labels_temp = local_train_labels_temp_1
            else:
                local_train_features_temp = np.concatenate((local_train_features_temp, local_train_features_temp_1), 0)
                local_train_labels_temp = np.concatenate((local_train_labels_temp, local_train_labels_temp_1), 0)
        ####shuffle the current local dataset##########################
        idx = np.arange(local_train_labels_temp.shape[0])
        np.random.shuffle(idx)  # randomize the idx
        local_train_features_temp = local_train_features_temp[idx]  # randomized dataset
        local_train_labels_temp = local_train_labels_temp[idx]  # corresponding training label
        ######local dataset pad##################
        local_train_features_temp = local_train_features_temp / 255.0
        ###################################################
        local_train_features_list.append(local_train_features_temp)
        local_train_labels_list.append(local_train_labels_temp)
        local_dataset_size.append(len(local_train_labels_temp))
        print('local_train_labels=',local_train_labels_temp)
        assert len(local_train_labels_temp) == shard_num_per_client*shard_size

    print('local_dataset_size=',local_dataset_size)

    return local_train_features_list, local_train_labels_list, local_dataset_size

def create_local_dataset_noniid_dirichlet(train_features, train_labels, num_clients, alpha):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    local_train_features_list = []
    local_train_labels_list = []
    local_dataset_size = []
    ####### sort the trainset in sequence ###########
    idx = np.argsort(train_labels)
    train_features = train_features[idx]  # randomized dataset
    train_labels = train_labels[idx]  # corresponding training label
    #################################################
    
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*num_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(num_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    assert len(client_idcs) == num_clients
    for i in range(num_clients):
        local_train_features_temp = train_features[client_idcs[i]]
        local_train_labels_temp = train_labels[client_idcs[i]]
        ####shuffle the current local dataset##########################
        idx = np.arange(local_train_labels_temp.shape[0])
        np.random.shuffle(idx)  # randomize the idx
        local_train_features_temp = local_train_features_temp[idx]  # randomized dataset
        local_train_labels_temp = local_train_labels_temp[idx]  # corresponding training label
        ######local dataset pad##################
        local_train_features_temp = local_train_features_temp / 255.0
        #########################################
        local_train_features_list.append(local_train_features_temp)
        local_train_labels_list.append(local_train_labels_temp)
        local_dataset_size.append(len(local_train_labels_temp))
        

    return local_train_features_list, local_train_labels_list, local_dataset_size



###############################$$$$$$$$$$$$$$$#######################
leaky_relu_alpha = 0.1

def conv2d_bayesian(inputs, filters_mu, filters_rho, stride_size, bias_b):
    filters_sigma = tf.math.softplus(filters_rho)  # standard deviation
    sigmab = tf.nn.conv2d(tf.math.square(inputs), tf.math.square(filters_sigma), strides=[1, stride_size, stride_size, 1], padding='SAME')
    tf.debugging.assert_non_negative(sigmab)
    out = tf.nn.conv2d(inputs, filters_mu, strides=[1, stride_size, stride_size, 1], padding='SAME') + bias_b + tf.random.normal(sigmab.shape) * tf.math.sqrt(tf.maximum(sigmab, 1e-10))

    return tf.nn.leaky_relu(out, alpha=leaky_relu_alpha)

def conv2d(inputs, filters_mu, stride_size, bias_b):
    out = tf.nn.conv2d(inputs, filters_mu, strides=[ 1 , stride_size , stride_size , 1 ] , padding='SAME') + bias_b

    return tf.nn.leaky_relu( out , alpha=leaky_relu_alpha )

def maxpool( inputs , pool_size , stride_size ):
    return tf.nn.max_pool2d( inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='VALID' , strides=[ 1 , stride_size , stride_size , 1 ] )

def dense_bayesian( inputs , weights_mu, weights_rho, bias_b ):
    weights_sigma = tf.math.softplus(weights_rho)  # standard deviation
    sigmab = tf.math.square(inputs) @ tf.cast(tf.math.square(weights_sigma), inputs.dtype)  # variance of activation (local reparametrization)
    temp_out = inputs @ weights_mu + bias_b + (tf.math.sqrt(tf.maximum(sigmab, 1e-10)) * tf.random.normal(sigmab.shape))
    x = tf.nn.leaky_relu( temp_out , alpha=leaky_relu_alpha )

    return x

def dense( inputs , weights_mu, bias_b ):
    temp_out = inputs @ weights_mu + bias_b
    x = tf.nn.leaky_relu( temp_out , alpha=leaky_relu_alpha )

    return x
#####################################################################
mu_initializer = tf.initializers.glorot_uniform()
# rho_initializer = tf.initializers.glorot_uniform()
# def mu_initializer(shape):
#     return tf.zeros(shape)

bias_initializer = tf.initializers.glorot_uniform()

def rho_initializer(shape):
    return -30.0 * tf.ones(shape)

# def rho_initializer(shape):
#     if tf.size(shape) == 4:
#         fan_in = shape[0] * shape[1] * shape[2]
#         fan_out = shape[0] * shape[1] * shape[3]
#     else:
#         fan_in = shape[0]
#         fan_out = shape[1]
#
#     return tfp.math.softplus_inverse(tf.sqrt((2/(fan_in+fan_out))*tf.ones(shape)))





def get_weight_mu( shape , name ):
    return tf.Variable( mu_initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

def get_weight_rho( shape , name ):
    return tf.Variable( rho_initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

def get_bias( shape , name ):
    return tf.Variable( bias_initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )


# ############################lenet###########################
# mu_shapes = [
#     [ 5 , 5 , 3 , 6 ] ,
#     [ 5 , 5 , 6 , 16 ] ,
#     [ 5 , 5 , 16 , 120 ] ,
#     [ 120 , 84 ] ,
#     [ 84 , 10 ] ,
# ]
#
# bias_shapes = [
#     [ 6 , ] ,
#     [ 16 , ] ,
#     [ 120 , ] ,
#     [ 84 , ] ,
#     [ 10 , ] ,
# ]
# ############################################################


mu_shapes = [
    [ 7 , 7 , 3 , 6 ] ,  # 32 32
    [ 5 , 5 , 6 , 16 ] ,  # 16 16
    [ 3 , 3 , 16 , 32 ] ,  # 8 8   4*4*32
    [ 3 , 3 , 32 , 64 ] ,  # 4 4
    # [3, 3, 16, 32],  # 2 2
    [ 576 , 120 ] ,
    [ 120 , 84 ] ,
    [ 84 , 10 ] ,
]

bias_shapes = [
    [ 6 , ] ,
    [ 16 , ] ,
    [ 32 , ] ,
    [ 64 , ] ,
    # [ 128 , ] ,
    [ 120 , ],
    [ 84 , ],
    [ 10 , ],
]
####################################initialize global weights_mu,rho,bias##############################
aggregated_weights_mu = []
aggregated_weights_rho = []
aggregated_bias = []

##############################################################################################
for i in range( len( mu_shapes ) ):
    # aggregated_weights_mu.append( get_weight_mu( mu_shapes[ i ] , 'weight_mu{}'.format( i ) ) )  # initialize weights_mu
    aggregated_weights_mu.append(tf.constant(get_weight_mu(mu_shapes[i], 'weight_mu{}'.format(i))))  # initialize weights_mu

for i in range(len(mu_shapes)):
    # aggregated_weights_rho.append(get_weight_rho(mu_shapes[i], 'weight_rho{}'.format(i)))  # initialize weights_rho
    aggregated_weights_rho.append(tf.constant(get_weight_rho(mu_shapes[i], 'weight_rho{}'.format(i))))  # initialize weights_rho

for i in range(len(bias_shapes)):
    # aggregated_bias.append(get_bias(bias_shapes[i], 'bias_b{}'.format(i)))  # initialize bias
    aggregated_bias.append(tf.constant(get_bias(bias_shapes[i], 'bias_b{}'.format(i))))  # initialize bias
###################################################################################
###################################################################
# �Զ���forwa����
# ############################lenet#############################
# def forward_run(x_in, weights_mu, weights_rho, bias_b):
#     x = tf.cast(x_in, dtype=tf.float32)
#     c1 = conv2d_bayesian(x, weights_mu[ 0 ], weights_rho[0], stride_size=1, bias_b=bias_b[0])
#     p1 = maxpool(c1, pool_size=2, stride_size=2)
#     c2 = conv2d_bayesian(p1, weights_mu[1], weights_rho[1], stride_size=1, bias_b=bias_b[1])
#     p2 = maxpool(c2, pool_size=2, stride_size=2)
#     c3 = conv2d_bayesian(p2, weights_mu[2], weights_rho[2], stride_size=1, bias_b=bias_b[2])
#     flatten = tf.reshape(c3, shape=(tf.shape(c3)[0], -1))
#     d1 = dense_bayesian(flatten, weights_mu[3], weights_rho[3], bias_b[3])
#     d2 = dense_bayesian(d1, weights_mu[4], weights_rho[4], bias_b[4])
#
#     return d2
#
# def forward_run_deter(x_in, weights_mu, bias_b):
#     x = tf.cast(x_in, dtype=tf.float32)
#     c1 = conv2d(x, weights_mu[0], stride_size=1, bias_b=bias_b[0])
#     p1 = maxpool(c1, pool_size=2, stride_size=2)
#     c2 = conv2d(p1, weights_mu[1], stride_size=1, bias_b=bias_b[1])
#     p2 = maxpool(c2, pool_size=2, stride_size=2)
#     c3 = conv2d(p2, weights_mu[2], stride_size=1, bias_b=bias_b[2])
#     flatten = tf.reshape(c3, shape=(tf.shape(c3)[0], -1))
#     d1 = dense(flatten, weights_mu[3], bias_b[3])
#     d2 = dense(d1, weights_mu[4], bias_b[4])
#
#     return d2
# ##############################################################
def forward_run(x_in, weights_mu, weights_rho, bias_b):
    x = tf.cast(x_in, dtype=tf.float32)
    c1 = conv2d_bayesian(x, weights_mu[ 0 ], weights_rho[0], stride_size=1, bias_b=bias_b[0])
    p1 = maxpool(c1, pool_size=3, stride_size=2)
    c2 = conv2d_bayesian(p1, weights_mu[1], weights_rho[1], stride_size=1, bias_b=bias_b[1])
    p2 = maxpool(c2, pool_size=3, stride_size=2)
    c3 = conv2d_bayesian(p2, weights_mu[2], weights_rho[2], stride_size=1, bias_b=bias_b[2])
    # p3 = maxpool(c3, pool_size=2, stride_size=2)
    c4 = conv2d_bayesian(c3, weights_mu[3], weights_rho[3], stride_size=1, bias_b=bias_b[3])
    p4 = maxpool(c4, pool_size=3, stride_size=2)
    # c5 = conv2d_bayesian(p4, weights_mu[4], weights_rho[4], stride_size=1, bias_b=bias_b[4])
    # p5 = maxpool(c5, pool_size=2, stride_size=2)
    flatten = tf.reshape(p4, shape=(tf.shape(p4)[0], -1))
    d1 = dense_bayesian(flatten, weights_mu[4], weights_rho[4], bias_b[4])
    d2 = dense_bayesian(d1, weights_mu[5], weights_rho[5], bias_b[5])
    d3 = dense_bayesian(d2, weights_mu[6], weights_rho[6], bias_b[6])

    return d3

def forward_run_deter(x_in, weights_mu, bias_b):
    x = tf.cast(x_in, dtype=tf.float32)
    c1 = conv2d(x, weights_mu[0], stride_size=1, bias_b=bias_b[0])   # output: 32*32*64   28*28
    p1 = maxpool(c1, pool_size=3, stride_size=2)   # output: 16*16*64                     14*14
    c2 = conv2d(p1, weights_mu[1], stride_size=1, bias_b=bias_b[1])  # output: 16*16*128  10*10
    p2 = maxpool(c2, pool_size=3, stride_size=2)  # output: 8*8*128                       5*5
    c3 = conv2d(p2, weights_mu[2], stride_size=1, bias_b=bias_b[2])  # output: 8*8      1*1
    # p3 = maxpool(c3, pool_size=2, stride_size=2)  # output: 4*4
    c4 = conv2d(c3, weights_mu[3], stride_size=1, bias_b=bias_b[3])  # output: 4*4
    p4 = maxpool(c4, pool_size=3, stride_size=2)  # output: 2*2
    # c5 = conv2d(p4, weights_mu[4], stride_size=1, bias_b=bias_b[4])  # output: 2*2
    # p5 = maxpool(c5, pool_size=2, stride_size=2)  # output: 1*1
    flatten = tf.reshape(p4, shape=(tf.shape(p4)[0], -1))
    d1 = dense(flatten, weights_mu[4], bias_b[4])
    d2 = dense(d1, weights_mu[5], bias_b[5])
    d3 = dense(d2, weights_mu[6], bias_b[6])

    return d3


def kl_loss(weights_rho, weights_mu, new_prior_sigma):
    kl_loss_val = 0
    for i in range(len(weights_rho)):
        sigma_temp = tf.math.softplus(weights_rho[i])
        if i == 0:
            prior_sigma_temp = tf.reshape(new_prior_sigma[:int(tf.math.reduce_prod(mu_shapes[i]))], sigma_temp.shape)
        else:
            prior_sigma_temp = tf.reshape(new_prior_sigma[sum([int(tf.math.reduce_prod(mu_shapes[j])) for j in range(i)]):sum([int(tf.math.reduce_prod(mu_shapes[j])) for j in range(i+1)])], sigma_temp.shape)
        pp_temp = tf.reduce_sum(tf.math.log(prior_sigma_temp / sigma_temp))
        tt_temp = tf.reduce_sum((tf.square(sigma_temp) + tf.square(weights_mu[i])) / (2 * tf.square(prior_sigma_temp)))
        kl_loss_val = kl_loss_val + pp_temp + tt_temp

    return kl_loss_val


def neg_log_likelihood(y_obs, y_pred):
    '''y_pred_temp = tf.reshape(y_pred, y_obs.shape)
    dist = tfp.distributions.Normal(loc=y_pred_temp, scale=sigma)  # regression'''
    cross_entropy_value = tf.keras.losses.sparse_categorical_crossentropy(y_obs, y_pred, from_logits=True)

    return tf.reduce_sum(tf.reduce_sum(cross_entropy_value))


def loss_fun(train_features, train_labels, weights_mu, weights_rho,
             prior_sigma, bias_b):
    prior_sigma_temp = tf.cast(prior_sigma, tf.float32)
    pred_labels = forward_run(train_features, weights_mu, weights_rho, bias_b)
    zp_neg_log_likelihood_real = neg_log_likelihood(train_labels, pred_labels)
    zp2 = kl_loss(weights_rho, weights_mu, prior_sigma_temp)

    return zp_neg_log_likelihood_real/32*50000 + 10 * zp2


# ��loss�ĺ���
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10000])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

# local pruning
def local_prunning(final_w3, no_prune_percent):
    final_w3_jj_list = []
    for i in range(len(final_w3)):
        w3_size = tf.size(final_w3[i]).numpy()
        k = int(no_prune_percent * w3_size)  # û��prune��Ԫ�ظ���
        top_values, top_indices = tf.nn.top_k(tf.reshape(tf.abs(final_w3[i]), (-1,)), k)
        top_value_min = tf.reduce_min(top_values)  # û��prune��Ԫ����abs��С��ֵ
        mask = tf.cast(tf.abs(final_w3[i]) >= top_value_min, tf.float32)
        final_w3_jj_list.append(tf.math.multiply(final_w3[i], mask))

    return final_w3_jj_list

# local pruning by magnitude
def local_prunning_mag(final_w3, prune_threshold_cov, prune_threshold):
    final_w3_jj_list = []
    for i in range(0,4):
        mask = tf.cast(tf.abs(final_w3[i]) >= prune_threshold_cov, tf.float32)
        # mask_new = tf.clip_by_value(mask, clip_value_min=tf.reduce_min(final_w3))
        final_w3_jj_list.append(tf.math.multiply(final_w3[i], mask))
    for i in range(4,len(final_w3)):
        mask = tf.cast(tf.abs(final_w3[i]) >= prune_threshold, tf.float32)
        # mask_new = tf.clip_by_value(mask, clip_value_min=tf.reduce_min(final_w3))
        final_w3_jj_list.append(tf.math.multiply(final_w3[i], mask))

    return final_w3_jj_list

# quantization function
def quantization(v):
    mask = tf.cast(v != 0, v.dtype)
    s_quantize = 2**8  # s is the number of quantization level
    l = tf.floor(tf.abs(v)/tf.norm(v)*s_quantize)  # l is a tensor the same shape as v
    ls_matrix = l/s_quantize  # l/s matrix, 1-p
    ls_matrix_2 = (l+1)/s_quantize  # (l+1)/s matrix, p
    prob_matrix = tf.abs(v)/tf.norm(v)*s_quantize - l  # probability p(a,s)
    counts = 1
    jvd = tfp.distributions.Binomial(counts, prob_matrix).sample()  # prob p
    jvd_inv = tf.cast(tf.abs(jvd) == 0, jvd.dtype)  # prob 1-p
    epsilon_1 = jvd_inv * ls_matrix
    epsilon_2 = jvd * ls_matrix_2
    epsilon = epsilon_1 + epsilon_2
    final_output = tf.norm(v) * tf.math.sign(v) * epsilon * mask

    return final_output

# quantization list
def quantization_list(v):
    final_output_list = []
    for i in range(len(v)):
        mask = tf.cast(v[i] != 0, v[i].dtype)
        s_quantize = 2**16  # s is the number of quantization level
        l = tf.floor(tf.abs(v[i])/tf.norm(v[i])*s_quantize)  # l is a tensor the same shape as v
        ls_matrix = l/s_quantize  # l/s matrix, 1-p
        ls_matrix_2 = (l+1)/s_quantize  # (l+1)/s matrix, p
        prob_matrix = tf.abs(v[i])/tf.norm(v[i])*s_quantize - l  # probability p(a,s)
        counts = 1
        jvd = tfp.distributions.Binomial(counts, prob_matrix).sample()  # prob p
        jvd_inv = tf.cast(tf.abs(jvd) == 0, jvd.dtype)  # prob 1-p
        epsilon_1 = jvd_inv * ls_matrix
        epsilon_2 = jvd * ls_matrix_2
        epsilon = epsilon_1 + epsilon_2
        final_output_list.append(tf.norm(v[i]) * tf.math.sign(v[i]) * epsilon * mask)

    return final_output_list

def count_bits_list(v):
    total_bits = 0
    for i in range(len(v)):
        value_bits = 16*tf.reduce_sum(tf.cast(v[i] != 0, v[i].dtype))
        location_bits = 2 * 8 * tf.reduce_sum(tf.cast(v[i] != 0, v[i].dtype))
        total_bits_this_layer = value_bits + location_bits
        total_bits = total_bits + total_bits_this_layer

    return total_bits

def calculate_sparsity_list(v):
    sparse_threshold = 0.01
    num_elements_gt = 0  # total number of nonzero elements in this model
    for i in range(len(v)):
        elements_gt = tf.math.greater_equal(abs(v[i]), sparse_threshold)
        num_elements_gt = num_elements_gt + tf.math.reduce_sum(tf.cast(elements_gt, tf.int32))

    sparsity_this_model = num_elements_gt / total_weight_number

    return sparsity_this_model


def aggregation_list(list, num_clients, local_dataset_size_list):  # for mu_weights, bias
    assert len(list) == num_clients == len(local_dataset_size_list)  # check list length == num_clients
    assert len(list[0]) == len(mu_shapes)  # check len of each subist == len of Lenet
    aggregated_list = list[0]  # assign client_0 value for the aggregated list, can be weights_mu, bias
    aggre_const_temp = local_dataset_size_list[0] / sum(local_dataset_size_list)
    for j in range(len(aggregated_list)):
        aggregated_list[j]=aggregated_list[j] * aggre_const_temp
    for i in range(1, num_clients):
        aggre_const_temp = local_dataset_size_list[i] / sum(local_dataset_size_list)
        if i % 5 == 0:
            print('aggregation_const:',aggre_const_temp)
        for j in range(len(aggregated_list)):
            aggregated_list[j]=(aggregated_list[j] + list[i][j] * aggre_const_temp)  # add weights of client_1, client_2, ...

    assert len(aggregated_list) == len(list[0])

    return aggregated_list


def aggregation_rho_list(list, num_clients, local_dataset_size_list):  # for mu_rho
    assert len(list) == num_clients == len(local_dataset_size_list)   # check list length == num_clients
    assert len(list[0]) == len(mu_shapes)  # check len of each subist == len of Lenet
    for i in range(len(list)):
        for j in range(len(list[0])):
            list[i][j]=(tf.square(tf.math.softplus(list[i][j])))  # convert weights_rho to  weights_variance

    aggregated_list = list[0]  # assign client_0 value for the aggregated list, can be weights_mu, bias
    aggre_const_temp = local_dataset_size_list[0] / sum(local_dataset_size_list)
    for j in range(len(aggregated_list)):
        aggregated_list[j]=aggregated_list[j] * aggre_const_temp
    for i in range(1, num_clients):
        aggre_const_temp = local_dataset_size_list[i] / sum(local_dataset_size_list)
        if i % 5 == 0:
            print('aggregation_const:',aggre_const_temp)
        for j in range(len(aggregated_list)):
            aggregated_list[j]=(aggregated_list[j] + list[i][j] * aggre_const_temp)  # add weights of client_1, client_2, ...

    assert len(aggregated_list) == len(list[0])

    return aggregated_list


# fine tuning
def forward_run_fine_tune(x_in, weights_mu, bias_b, mask):
    x = tf.cast(x_in, dtype=tf.float32)
    c1 = conv2d(x, mask[0] * weights_mu[0], stride_size=1, bias_b=bias_b[0])
    p1 = maxpool(c1, pool_size=2, stride_size=2)
    c2 = conv2d(p1, mask[1] * weights_mu[1], stride_size=1, bias_b=bias_b[1])
    p2 = maxpool(c2, pool_size=2, stride_size=2)
    c3 = conv2d(p2, mask[2] * weights_mu[2], stride_size=1, bias_b=bias_b[2])
    flatten = tf.reshape(c3, shape=(tf.shape(c3)[0], -1))
    d1 = dense(flatten, mask[3] * weights_mu[3], bias_b[3])
    d2 = dense(d1, mask[4] * weights_mu[4], bias_b[4])

    return d2

def loss_fine_tune(train_features, train_labels, weights_mu, bias_b, mask):
    pred_labels = forward_run_fine_tune(train_features, weights_mu, bias_b, mask)
    zp_neg_log_likelihood_real = neg_log_likelihood(train_labels, pred_labels)

    return zp_neg_log_likelihood_real/32*500

def fine_tune(l_rate, epoch_num, batch_size, train_size_, train_features, train_labels, weights_mu_pruned, bias_b, test_features, test_labels):
    loss_temp = []
    mask = []
    for i in range(len(weights_mu_pruned)):
        mask.append(tf.cast(weights_mu_pruned[i] != 0, weights_mu_pruned[i].dtype))

    weights_mu_local = []
    bias_b_local = []

    for i in range(len(weights_mu_pruned)):
        weights_mu_local.append(tf.Variable(weights_mu_pruned[i]))
        bias_b_local.append(tf.Variable(bias_b[i]))

    optimizer = tf.optimizers.Adam(learning_rate=l_rate)
    num_batches = math.ceil(train_size_ / batch_size)
    for i in range(epoch_num):
        for j in range(num_batches):
            # train_features_batch = train_features[range(j * batch_size, (j + 1) * batch_size)]
            train_features_batch = train_features[j * batch_size: (j + 1) * batch_size]
            # train_labels_batch = train_labels[range(j * batch_size, (j + 1) * batch_size)]
            train_labels_batch = train_labels[j * batch_size: (j + 1) * batch_size]
            with tf.GradientTape() as g:
                loss_value = loss_fine_tune(train_features_batch, train_labels_batch, weights_mu_local, bias_b_local, mask)

            trainable_variables = weights_mu_local + bias_b_local

            gradients = g.gradient(loss_value, trainable_variables)  # ��ȡ�ݶ�

            optimizer.apply_gradients(zip(gradients, trainable_variables))

            # loss_temp.append(loss_value)
            # print(loss_value)
            # print(tf.norm(gradients[2], ord=2))
            # gradddd.append(tf.norm(gradients[2], ord=2))

        loss_temp.append(loss_value)

    plt.plot(loss_temp)
    plt.show()

    #######################################test accuracy########################################################
    test_predictions = forward_run_deter(test_features, weights_mu_local, bias_b_local)  # �õ�test����Predictions

    pred_error = tf.keras.losses.sparse_categorical_crossentropy(test_labels, test_predictions,
                                                                 from_logits=True)  # �õ�pred_error
    print('*****fine_tune_pred_error:', pred_error)
    prediction_probability = tf.nn.softmax(test_predictions)
    test_predictions_labels = tf.argmax(prediction_probability, axis=1)
    accuracy = tf.reduce_sum(
        tf.cast(tf.equal(test_predictions_labels, test_labels), test_predictions_labels.dtype)) / len(test_labels)
    print('Fine_tune_ACCURACY:', accuracy)

    return weights_mu_local, bias_b_local, mask, accuracy

batch_size = 128
it_num = 100  # �ܵ�������
# total_weight_number = int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])+tf.math.reduce_prod(mu_shapes[5])+tf.math.reduce_prod(mu_shapes[6]))
total_weight_number = int(sum([tf.math.reduce_prod(mu_shapes[i]) for i in range(len(mu_shapes))]))
a = tf.Variable(1.0 * np.ones(total_weight_number), dtype=tf.double)  # a��ʼֵ
a_bar = tf.Variable(1.0 * np.ones(total_weight_number), dtype=tf.double)  # a_bar��ʼֵ
b = tf.Variable(1. * np.ones(total_weight_number), dtype=tf.double)  # b��ʼֵ
b_bar = tf.Variable(0.01 * np.ones(total_weight_number), dtype=tf.double)  # b_bar��ʼֵ 0.0001
a_tilde = tf.Variable(1. * np.ones(total_weight_number), dtype=tf.double)  # 1
b_tilde = tf.Variable(1. * np.ones(total_weight_number), dtype=tf.double)  # 0.01

# pi_tilde = tf.Variable(np.random.rand(total_weight_number), dtype=tf.double)  # ��ʼ��pi-tilde����VBI�������q��s���ĸ��ʣ�����SPMP
# pi_tilde_kuozhan = tf.Variable(np.random.rand(total_weight_number), dtype=tf.double)  # pi_tilde��ʼֵ
pi_tilde = tf.Variable(0.5*np.ones(total_weight_number), dtype=tf.double)
pi_tilde_kuozhan = tf.Variable(0.5*np.ones(total_weight_number), dtype=tf.double)

new_prior_sigma = tf.Variable(1.0 * np.ones(total_weight_number))
new_prior_rho = tf.Variable(0.5 * np.random.rand(total_weight_number))
temp1 = tf.Variable(np.ones(total_weight_number), dtype=tf.double)
temp2 = tf.Variable(np.ones(total_weight_number), dtype=tf.double)
a_tilde_old = tf.Variable(1.25 * np.ones(total_weight_number), dtype=tf.double)
b_tilde_old = tf.Variable(0.5 * np.ones(total_weight_number), dtype=tf.double)
pi_tilde_old = tf.Variable(np.zeros(total_weight_number), dtype=tf.double)

# pi_np = np.ones(total_weight_number)
# pi_np[5*5*1*6+5*5*6*16+5*5*16*120:5*5*1*6+5*5*6*16+5*5*16*120+60*84] = 0.0
# pi = tf.Variable(pi_np)
# pi_read = pd.read_csv("pi_smallalex.csv")
# pi_read_tensor = tf.concat([[1.0], tf.reshape(tf.convert_to_tensor(pi_read), -1)],0)
# pi = tf.Variable(pi_read_tensor, dtype=tf.double)  # ��ʼ��pi, ��SPMP����ı�Ե���ʣ�����VBI
# print('pi = ', pi)
pi = tf.Variable(0.5*np.ones(total_weight_number), dtype=tf.double)  # ��ʼ��pi, ��SPMP����ı�Ե���ʣ�����VBI
pi_old = tf.Variable(0.5*np.random.rand(total_weight_number), dtype=tf.double)
# pi = tf.Variable(np.random.rand(nn_param_num), dtype=tf.double)  # ��ʼ��pi, ��SPMP����ı�Ե���ʣ�����VBI

pi_error = []  #��ѭ������������
pi_error.append(np.linalg.norm(pi - pi_old) / np.linalg.norm(pi))
ppv = 1
l_rate = 1e-3
def local_train(l_rate, batch_size_, train_size_, train_features,
                    train_labels, weights_mu,
                weights_rho, bias_b, test_features, test_labels, new_prior_sigma):
    train_iterat = 5  # 15
    l_rate_local = l_rate
    batch_size = batch_size_
    num_batches = math.ceil(train_size_ / batch_size)
    weights_mu_local = []
    weights_rho_local = []
    bias_b_local = []
    for i in range(len(weights_mu)):
        weights_mu_local.append(tf.Variable(weights_mu[i], name='weights_mu'))
        weights_rho_local.append(tf.Variable(weights_rho[i], name='weights_rho'))
        bias_b_local.append(tf.Variable(bias_b[i], name='bias'))

    new_prior_sigma_local = tf.Variable(new_prior_sigma)  # ��weights��ʼ��prior_sigma # neuron_num1 for 20 neurons

    loss = []  # ����ÿ�δ������loss
    # optimizer
    optimizer = tf.optimizers.Adam(learning_rate=l_rate_local)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=l_rate_local, momentum=0.9)
    loss_temp = []
    # ����ѵ������
    for i in range(train_iterat):
        for j in range(num_batches):
            # train_features_batch = train_features[range(j * batch_size, (j + 1) * batch_size)]
            train_features_batch = train_features[j * batch_size: (j + 1) * batch_size]
            # train_labels_batch = train_labels[range(j * batch_size, (j + 1) * batch_size)]
            train_labels_batch = train_labels[j * batch_size: (j + 1) * batch_size]
            with tf.GradientTape() as g:
                loss_value = loss_fun(train_features_batch, train_labels_batch, weights_mu_local,
                                      weights_rho_local, new_prior_sigma_local, bias_b_local)

            # trainable_variables = weights_mu_local + weights_rho_local + bias_b_local
            trainable_variables = weights_mu_local + bias_b_local

            gradients = g.gradient(loss_value, trainable_variables)  # ��ȡ�ݶ�
            # gradients = [(tf.clip_by_norm(grad,5.0)) for grad in gradients]
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            # loss_temp.append(loss_value)
            # print(loss_value)
            # print(tf.norm(gradients[4], ord=2))
            # gradddd.append(tf.norm(gradients[2], ord=2))

        loss_temp.append(loss_value)

    loss.append(loss_value)  # ����10������ǰ��loss
    print(loss)

    # for l in range(4,7):  # see how many neurons are shut down in the fc layers (layer 2,3,4)
    #     print(np.sum(np.sum(np.sum(np.abs(trainable_variables[l]), axis=1) > 10 ** -3)))
    #     print(np.where(np.sum(np.abs(trainable_variables[l]), axis=1) < 10 ** -3))

    #######################################test accuracy########################################################
    accuracy_sum = 0
    for ib in range(100):  # divide test set into 100 batches
        test_predictions = forward_run_deter(test_features[ib * 100: (ib + 1) * 100], weights_mu_local,
                                             bias_b_local)  # �õ�test����Predictions

        prediction_probability = tf.nn.softmax(test_predictions)
        test_predictions_labels = tf.argmax(prediction_probability, axis=1)
        accuracy = tf.reduce_sum(
            tf.cast(tf.equal(test_predictions_labels, test_labels[ib * 100: (ib + 1) * 100]),
                    test_predictions_labels.dtype)) / len(test_labels[ib * 100: (ib + 1) * 100])
        accuracy_sum = accuracy_sum + accuracy

    accuracy_avg = accuracy_sum / 100
    print('Test_ACC_this_client:', accuracy_avg)

    #####################--prunning--#################################
    # prune_threshold = 0.05
    # weights_mu_local = local_prunning_mag(weights_mu_local, prune_threshold)

    weights_mu_local_const = []  # convert to tf.constant
    weights_rho_local_const = []
    bias_b_local_const = []
    for i in range(len(weights_mu_local)):
        weights_mu_local_const.append(tf.constant(weights_mu_local[i]))
        weights_rho_local_const.append(tf.constant(weights_rho_local[i]))
        bias_b_local_const.append(tf.constant(bias_b_local[i]))

    return accuracy_avg, weights_mu_local_const, weights_rho_local_const, bias_b_local_const


###########################CREATE LOCAL DATASET###########################################
num_clients = 50  # create local dataset
local_train_features_list, local_train_labels_list, local_dataset_size = create_local_dataset(train_features, train_labels, num_clients)
for i in range(5):
    plt.hist(local_train_labels_list[i])
    plt.show()
print('local dataset generated')
######################################################################
accuracy_global = []
a_tilde_error_list = []
b_tilde_error_list = []
pi_tilde_error_list = []

################################Module A############################################################################
a_tilde_error = 1
b_tilde_error = 1
pi_tilde_error = 1
u = 1
weights_mu_local_list = []  # store weights_mu from each client
weights_rho_local_list = []  # store weights_rho from each client
bias_b_local_list = []  # store bias_b from each client

total_bits = 0
total_bits_list = []
up_load_bits = 0
up_load_bits_list = []
avg_local_sparsity = 0
avg_local_sparsity_list = []
temp_sum_sparsity = 0
###############################################
for i in range(num_clients):  # each client perform w-step
    accuracy, weights_mu_local, weights_rho_local, bias_b_local = local_train(l_rate, batch_size, local_dataset_size[i], local_train_features_list[i],
                local_train_labels_list[i], aggregated_weights_mu,
                aggregated_weights_rho, aggregated_bias, test_features, test_labels,
                new_prior_sigma)

    ############################################
    weights_mu_local_pruned = local_prunning_mag(weights_mu_local, 0.01, 0.01)
    weights_mu_local_quantized = quantization_list(weights_mu_local_pruned)
    weights_mu_local_list.append(weights_mu_local_pruned)
    weights_rho_local_list.append(weights_rho_local)
    bias_b_local_list.append(bias_b_local)
    print('local_accuracy_this_client,', accuracy)

    bits_this_client = count_bits_list(weights_mu_local_pruned)
    up_load_bits = up_load_bits + bits_this_client

    sparsity_this_client = calculate_sparsity_list(weights_mu_local_pruned)
    temp_sum_sparsity = temp_sum_sparsity + sparsity_this_client

avg_local_sparsity = temp_sum_sparsity/num_clients
avg_local_sparsity_list.append(avg_local_sparsity)
up_load_bits_list.append(up_load_bits)


down_load_bits_list = []
## aggregation###
aggregated_weights_mu = aggregation_list(weights_mu_local_list, num_clients, local_dataset_size)  # aggregated weights_mu
aggregated_weights_var = aggregation_rho_list(weights_rho_local_list, num_clients, local_dataset_size)  # aggregated weights_variance
aggregated_weights_rho = [0 for i in range(len(aggregated_weights_var))]  # construct aggregated rho list
for i in range(len(aggregated_weights_var)):
    assert len(aggregated_weights_var) == len(aggregated_weights_rho)  # assert aggregated var length == aggregated rho list
    aggregated_weights_rho[i] = tf.Variable(tfp.math.softplus_inverse(tf.sqrt(aggregated_weights_var[i])), name='weights_rho')  # calculate aggregateed rho according to aggregated variance; need to be feedback to clients

aggregated_bias = aggregation_list(bias_b_local_list, num_clients, local_dataset_size)  # aggregated bias

aggregated_weights_mu_flat = [0 for i in range(len(aggregated_weights_mu))]  # ����һ����list�����������ƽ���
for i in range(len(aggregated_weights_mu)):
    aggregated_weights_mu_flat[i] = tf.reshape(aggregated_weights_mu[i], [-1])

aggregated_weights_var_flat = [0 for i in range(len(aggregated_weights_var))]  # ����һ����list�����������ƽ���
for i in range(len(aggregated_weights_var)):
    aggregated_weights_var_flat[i] = tf.reshape(aggregated_weights_var[i], [-1])

trainable_variable_mu = tf.concat(
        [aggregated_weights_mu_flat[0], aggregated_weights_mu_flat[1], aggregated_weights_mu_flat[2],
         aggregated_weights_mu_flat[3], aggregated_weights_mu_flat[4], aggregated_weights_mu_flat[5], aggregated_weights_mu_flat[6]], 0)

trainable_variable_var = tf.concat(
    [aggregated_weights_var_flat[0], aggregated_weights_var_flat[1], aggregated_weights_var_flat[2],
     aggregated_weights_var_flat[3], aggregated_weights_var_flat[4], aggregated_weights_var_flat[5], aggregated_weights_var_flat[6]], 0)


#########################################count bits#######################
down_load_bits = count_bits_list(aggregated_weights_mu)/2*3
down_load_bits_list.append(down_load_bits)
total_bits = down_load_bits + up_load_bits
total_bits_list.append(total_bits)
#######################################global test accuracy########################################################
accuracy_sum = 0
for ib in range(100):  # divide test set into 100 batches
    test_predictions = forward_run_deter(test_features[ib*100 : (ib+1)*100], aggregated_weights_mu, aggregated_bias)  # �õ�test����Predictions

    pred_error = tf.keras.losses.sparse_categorical_crossentropy(test_labels[ib*100 : (ib+1)*100], test_predictions, from_logits=True)  # �õ�pred_error
    # print('*****pred_error:', pred_error)
    prediction_probability = tf.nn.softmax(test_predictions)
    test_predictions_labels = tf.argmax(prediction_probability, axis=1)
    accuracy = tf.reduce_sum(
        tf.cast(tf.equal(test_predictions_labels, test_labels[ib*100 : (ib+1)*100]), test_predictions_labels.dtype)) / len(test_labels[ib*100 : (ib+1)*100])
    accuracy_sum = accuracy_sum + accuracy

accuracy_avg = accuracy_sum / 100
print('GLOBAL_ACCURACY:', accuracy_avg)
accuracy_global.append(accuracy_avg)
# plt.plot(accuracy_global)
# plt.show()
##########################################################################################################
############################################################rho-step##################################################
# rho�ֲ�����
a_tilde.assign(pi_tilde_kuozhan * a + (1 - pi_tilde_kuozhan) * a_bar + np.ones(len(trainable_variable_mu)))
b_tilde.assign(tf.cast(
    tf.square(trainable_variable_mu) + trainable_variable_var + tf.cast(pi_tilde_kuozhan * b,
                                                                              dtype=tf.float32) + tf.cast(
        (1 - pi_tilde_kuozhan) * b_bar, dtype=tf.float32), dtype=tf.double))
new_prior_rho.assign(a_tilde / b_tilde)
print(tf.norm(new_prior_sigma))
new_prior_sigma.assign(1.0 / np.sqrt(new_prior_rho))  # need to be feedback to clients
print(tf.norm(new_prior_sigma))
a_tilde_error = tf.norm(tf.abs(a_tilde - a_tilde_old)) / tf.norm(tf.abs(a_tilde))
b_tilde_error = tf.norm(tf.abs(b_tilde - b_tilde_old)) / tf.norm(tf.abs(b_tilde))
if np.all(a_tilde - a_tilde_old) < 0.008:
    a_tilde_error = 1e-6
if np.all(b_tilde - b_tilde_old) < 0.008:
    b_tilde_error = 1e-6
a_tilde_old.assign(a_tilde)
b_tilde_old.assign(b_tilde)
############################################################s-step#####################################################
# s�ֲ�����
temp1.assign(pi * (b ** a) / tf.math.exp(tf.math.lgamma(a)) * tf.math.exp(
    (a - 1.0) * (tf.math.digamma(a_tilde) - tf.math.log(b_tilde)) - b * (
                a_tilde / b_tilde)))
temp1.assign(tf.clip_by_value(temp1, 1e-10, 1e1000000))
temp2.assign(
    (1.0 - pi) * (b_bar ** a_bar) / tf.math.exp(tf.math.lgamma(a_bar)) * tf.math.exp(
        (a_bar - 1.0) * (tf.math.digamma(a_tilde) - tf.math.log(b_tilde)) - b_bar * (
                    a_tilde / b_tilde)))
pi_tilde.assign(temp1 / (temp1 + temp2))

pi_tilde_kuozhan.assign(pi_tilde)  # ��pi_tilde_kuozhan��ֵ

pi_tilde_error = tf.norm(pi_tilde - pi_tilde_old) / tf.norm(pi_tilde)
print('pi_tilde_error=', pi_tilde_error)
if np.all(pi_tilde - pi_tilde_old) < 1e-6:
    pi_tilde_error_local = 1e-6  # ############################################################## ����nan��������������
pi_tilde_old.assign(pi_tilde)
###############################plot figures#####################################
pi_tilde_dense_1 = np.reshape(pi_tilde[int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])):int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])+tf.math.reduce_prod(mu_shapes[5]))],
                              (120, 84))

sns_pi_tilde_dense_1 = sns.heatmap(pi_tilde_dense_1)
plt.title("pi_tilde_dense_1")
plt.show()
############################################################################
pi_dense_1 = np.reshape(pi[int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])):int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])+tf.math.reduce_prod(mu_shapes[5]))],
                        (120, 84))

sns_pi_dense_1 = sns.heatmap(pi_dense_1)
plt.title("pi_dense_1")
plt.show()
########################################################################
pi_tilde_conv1 = tf.reshape(pi_tilde[:882], [7, 7, 3, 6])
pi_conv1 = tf.reshape(pi[:882], [7, 7, 3, 6])
aggregated_w_mu_dense_1 = aggregated_weights_mu[5]
aggregated_w_mu_conv1 = aggregated_weights_mu[0]

sns_aggregated_w_mu_dense1 = sns.heatmap(tf.clip_by_value(abs(aggregated_w_mu_dense_1), 0, 3))
plt.title("aggregated_w_mu_dense_1")
plt.show()

fig, ax = plt.subplots(3, 6)  # ÿ�зֱ�conv_1��weights_mu, pi, jbjb
for i in range(6):
    sns.heatmap(tf.abs(aggregated_w_mu_conv1[:, :, 0, i]), ax=ax[0][i])
    sns.heatmap(pi_conv1[:, :, 0, i], ax=ax[1][i])

plt.show()
############################################plot figure end############################################
print('a_tilde_error', a_tilde_error)
print('b_tilde_error', b_tilde_error)
print('pi_tilde_error', pi_tilde_error)
print(pi_tilde)
a_tilde_error_list.append(a_tilde_error)
b_tilde_error_list.append(b_tilde_error)
pi_tilde_error_list.append(pi_tilde_error)

u = u + 1
######end of Module A##################

sparsity_per_round = []
sparsity_per_round_ind = []
l_rate = 1e-3
bits_accumulated = 0
bits_per_round = []
ppv = 1
for i in range(60):  # module A Module B iteration number
    #############################################################Module B##########################################################
    print('Module B begins')
    # ͨ��SPMP�㷨��pi
    p01_row = 0.288  # 0.288  ###############for kernel###################
    p11_row = 0.49  # 0.49
    p01_col = 0.288  # 0.288
    p11_col = 0.49  # 0.49
    prob_start = np.array([0.3, 1 - 0.3])  # define start prob #### 0.0005
    pi_old.assign(pi)

    # start SPMP
    # conv_1, [7,7,3,6]
    row_num = 7*3  # row number of kernel
    col_num = 7*6  # col number of kernel

    g = FactorGraph(silent=True)  # init the graph without message printouts

    x = []  # list of vn (variable node)
    f_unary = []  # unary factor node
    f_start = []  # unary factor node at the beginning of matrix #####

    prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                        [p01_row, 1 - p01_row]])  # distribution of pairwise horizontal factor node
    f_pairwise_horizontal = []  # horizontal pairwise factor node

    prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                    [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
    f_pairwise_vertical = []  # vertical pairwise factor node
    for i in range(row_num * col_num):
        x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
        # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[j+i*3*6]]*(1-pi[j+i*3*6]]) / (pi[j+i*3*6]]+(1-2*pi[j+i*3*6]])*pi_tilde[j+i*3*6]]), 1 - pi_tilde[j+i*3*6]]*(1-pi[j+i*3*6]]) / (pi[j+i*3*6]]+(1-2*pi[j+i*3*6]])*pi_tilde[j+i*3*6]])])))  # define unary factor node
        f_unary.append(Factor('f_unary' + '{}'.format(i), np.array(
            [pi_tilde[i] * pi[i] / (
                        1 - (pi[i] + (1 - 2 * pi[i]) * pi_tilde[i])),
            1 - pi_tilde[i] * pi[i] / (1 - (
                        pi[i] + (1 - 2 * pi[i]) * pi_tilde[
                    i]))])))  # define unary factor node

        if i in range(row_num * (col_num - 1)):
            f_pairwise_horizontal.append(Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

        if i in range(col_num * (row_num - 1)):
            f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

        # if i < col_num or i % col_num == 0:
        #     f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

    # connect the vn and fns
    for i in range(row_num * col_num):
        g.add(x[i])
        g.add(f_unary[i])
        g.append('f_unary' + '{}'.format(i), x[i])

        # if i < col_num:  ####
        #     g.append('x' + '{}'.format(i), f_start[i])
        # elif i % col_num == 0:
        #     g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

        if (i % col_num) != (col_num - 1):  # if not the vn of the last column
            g.append('x' + '{}'.format(i),
                    f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

        if (i % col_num) != 0:  # if not the vn of the first column
            g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                    x[i])  # edge between each vn and its left-hand fn

        if i < (row_num * col_num - col_num):  # if not the vn of the last row
            g.append('x' + '{}'.format(i), f_pairwise_vertical[
                int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

        if i > (col_num - 1):  # if not the vn of the first row
            g.append('f_pairwise_vertical' + '{}'.format(
                int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                    x[i])  # edge between each vn and its above-hand fn

    g.compute_marginals()
    for i in range(row_num * col_num):
        # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[j+i*3*6]]) / tf.reduce_sum(g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[j+i*3*6]], 1 - pi_tilde[j+i*3*6]]])
        pi[i].assign(g.nodes['x' + '{}'.format(i)].marginal()[0])
        # pi[j+i*3*6]].assign(message_real)

    # cov_2 [5,5,6,16], 6*16�� 3*3 matrix
    row_num = 5*6  # row number of kernel
    col_num = 5*16  # col number of kernel

    g = FactorGraph(silent=True)  # init the graph without message printouts

    x = []  # list of vn (variable node)
    f_unary = []  # unary factor node
    f_start = []  # unary factor node at the beginning of matrix #####

    prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                        [p01_row, 1 - p01_row]])  # distribution of pairwise horizontal factor node
    f_pairwise_horizontal = []  # horizontal pairwise factor node

    prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                    [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
    f_pairwise_vertical = []  # vertical pairwise factor node
    for i in range(row_num * col_num):
        x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
        # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[j+i*6*16+3*3*3*6] * (1 - pi[j+i*6*16+3*3*3*6]) / (pi[j+i*6*16+3*3*3*6] + (1 - 2 * pi[j+i*6*16+3*3*3*6]) * pi_tilde[j+i*6*16+3*3*3*6]), 1 - pi_tilde[j+i*6*16+3*3*3*6] * (1 - pi[j+i*6*16+3*3*3*6]) / (pi[j+i*6*16+3*3*3*6] + (1 - 2 * pi[j+i*6*16+3*3*3*6]) * pi_tilde[j+i*6*16+3*3*3*6])])))  # define unary factor node
        f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[i + 7 * 7 * 3 * 6] *
                                                                    pi[i + 7 * 7 * 3 * 6] / (1 - (
                    pi[i + 7 * 7 * 3 * 6] + (
                    1 - 2 * pi[i + 7 * 7 * 3 * 6]) * pi_tilde[i + 7 * 7 * 3 * 6])),
                                                                    1 - pi_tilde[i + 7 * 7 * 3 * 6] *
                                                                    pi[i + 7 * 7 * 3 * 6] / (1 - (
                                                                            pi[i + 7 * 7 * 3 * 6] + (
                                                                            1 - 2 * pi[
                                                                        i + 7 * 7 * 3 * 6]) *
                                                                            pi_tilde[
                                                                                i + 7 * 7 * 3 * 6]))])))  # define unary factor node

        if i in range(row_num * (col_num - 1)):
            f_pairwise_horizontal.append(Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

        if i in range(col_num * (row_num - 1)):
            f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

        # if i < col_num or i % col_num == 0:
        #     f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

    # connect the vn and fns
    for i in range(row_num * col_num):
        g.add(x[i])
        g.add(f_unary[i])
        g.append('f_unary' + '{}'.format(i), x[i])

        # if i < col_num:  ####
        #     g.append('x' + '{}'.format(i), f_start[i])
        # elif i % col_num == 0:
        #     g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

        if (i % col_num) != (col_num - 1):  # if not the vn of the last column
            g.append('x' + '{}'.format(i),
                    f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

        if (i % col_num) != 0:  # if not the vn of the first column
            g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                    x[i])  # edge between each vn and its left-hand fn

        if i < (row_num * col_num - col_num):  # if not the vn of the last row
            g.append('x' + '{}'.format(i), f_pairwise_vertical[
                int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

        if i > (col_num - 1):  # if not the vn of the first row
            g.append('f_pairwise_vertical' + '{}'.format(
                int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                    x[i])  # edge between each vn and its above-hand fn

    g.compute_marginals()
    for i in range(row_num * col_num):
        # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[j+i*6*16+3*3*3*6]) / tf.reduce_sum(g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[j+i*6*16+3*3*3*6], 1 - pi_tilde[j+i*6*16+3*3*3*6]])
        pi[i + 7 * 7 * 3 * 6].assign(g.nodes['x' + '{}'.format(i)].marginal()[0])
        # pi[j+i*6*16+3*3*3*6].assign(message_real)

    # cov_3 [3,3,16,32], 16*32�� 3*3 matrix
    row_num = 3*16  # row number of kernel
    col_num = 3*32  # col number of kernel

    g = FactorGraph(silent=True)  # init the graph without message printouts

    x = []  # list of vn (variable node)
    f_unary = []  # unary factor node
    f_start = []  # unary factor node at the beginning of matrix #####

    prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                        [p01_row, 1 - p01_row]])  # distribution of pairwise horizontal factor node
    f_pairwise_horizontal = []  # horizontal pairwise factor node

    prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                    [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
    f_pairwise_vertical = []  # vertical pairwise factor node
    for i in range(row_num * col_num):
        x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
        # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[j+i*6*16+3*3*6] * (1 - pi[j+i*6*16+3*3*6]) / (pi[j+i*6*16+3*3*6] + (1 - 2 * pi[j+i*6*16+3*3*6]) * pi_tilde[j+i*6*16+3*3*6]), 1 - pi_tilde[j+i*6*16+3*3*6] * (1 - pi[j+i*6*16+3*3*6]) / (pi[j+i*6*16+3*3*6] + (1 - 2 * pi[j+i*6*16+3*3*6]) * pi_tilde[j+i*6*16+3*3*6])])))  # define unary factor node
        f_unary.append(Factor('f_unary' + '{}'.format(i),
                            np.array([pi_tilde[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16] *
                                        pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16] / (1 - (
                                    pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16] + (
                                    1 - 2 * pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16]) * pi_tilde[
                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16])),
                                        1 - pi_tilde[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16] *
                                        pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16] / (1 - (
                                                pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16] + (
                                                1 - 2 * pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16]) *
                                                pi_tilde[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16]))])))  # define unary factor node

        if i in range(row_num * (col_num - 1)):
            f_pairwise_horizontal.append(Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

        if i in range(col_num * (row_num - 1)):
            f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

        # if i < col_num or i % col_num == 0:
        #     f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

    # connect the vn and fns
    for i in range(row_num * col_num):
        g.add(x[i])
        g.add(f_unary[i])
        g.append('f_unary' + '{}'.format(i), x[i])

        # if i < col_num:  ####
        #     g.append('x' + '{}'.format(i), f_start[i])
        # elif i % col_num == 0:
        #     g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

        if (i % col_num) != (col_num - 1):  # if not the vn of the last column
            g.append('x' + '{}'.format(i),
                    f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

        if (i % col_num) != 0:  # if not the vn of the first column
            g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                    x[i])  # edge between each vn and its left-hand fn

        if i < (row_num * col_num - col_num):  # if not the vn of the last row
            g.append('x' + '{}'.format(i), f_pairwise_vertical[
                int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

        if i > (col_num - 1):  # if not the vn of the first row
            g.append('f_pairwise_vertical' + '{}'.format(
                int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                    x[i])  # edge between each vn and its above-hand fn

    g.compute_marginals()
    for i in range(row_num * col_num):
        # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[j+i*6*16+3*3*6]) / tf.reduce_sum(g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[j+i*6*16+3*3*6], 1 - pi_tilde[j+i*6*16+3*3*6]])
        pi[i+ 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16].assign(g.nodes['x' + '{}'.format(i)].marginal()[0])
        # pi[j+i*16*32+3*3*3*6+3*3*6*16].assign(message_real)

    # cov_4 [3,3,32,64], 32*64�� 3*3 matrix
    row_num = 3*32  # row number of kernel
    col_num = 3*64  # col number of kernel

    g = FactorGraph(silent=True)  # init the graph without message printouts

    x = []  # list of vn (variable node)
    f_unary = []  # unary factor node
    f_start = []  # unary factor node at the beginning of matrix #####

    prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                        [p01_row,
                                        1 - p01_row]])  # distribution of pairwise horizontal factor node
    f_pairwise_horizontal = []  # horizontal pairwise factor node

    prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                    [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
    f_pairwise_vertical = []  # vertical pairwise factor node
    for i in range(row_num * col_num):
        x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
        # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[j+i*6*16+3*3*6] * (1 - pi[j+i*6*16+3*3*6]) / (pi[j+i*6*16+3*3*6] + (1 - 2 * pi[j+i*6*16+3*3*6]) * pi_tilde[j+i*6*16+3*3*6]), 1 - pi_tilde[j+i*6*16+3*3*6] * (1 - pi[j+i*6*16+3*3*6]) / (pi[j+i*6*16+3*3*6] + (1 - 2 * pi[j+i*6*16+3*3*6]) * pi_tilde[j+i*6*16+3*3*6])])))  # define unary factor node
        f_unary.append(Factor('f_unary' + '{}'.format(i),
                            np.array(
                                [pi_tilde[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32] *
                                pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32] / (1 - (
                                        pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32] + (
                                        1 - 2 * pi[
                                    i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32]) *
                                        pi_tilde[
                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32])),
                                1 - pi_tilde[
                                    i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32] *
                                pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32] / (1 - (
                                        pi[
                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32] + (
                                                1 - 2 * pi[
                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32]) *
                                        pi_tilde[
                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32]))])))  # define unary factor node

        if i in range(row_num * (col_num - 1)):
            f_pairwise_horizontal.append(
                Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

        if i in range(col_num * (row_num - 1)):
            f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

        # if i < col_num or i % col_num == 0:
        #     f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

    # connect the vn and fns
    for i in range(row_num * col_num):
        g.add(x[i])
        g.add(f_unary[i])
        g.append('f_unary' + '{}'.format(i), x[i])

        # if i < col_num:  ####
        #     g.append('x' + '{}'.format(i), f_start[i])
        # elif i % col_num == 0:
        #     g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

        if (i % col_num) != (col_num - 1):  # if not the vn of the last column
            g.append('x' + '{}'.format(i),
                    f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

        if (i % col_num) != 0:  # if not the vn of the first column
            g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                    x[i])  # edge between each vn and its left-hand fn

        if i < (row_num * col_num - col_num):  # if not the vn of the last row
            g.append('x' + '{}'.format(i), f_pairwise_vertical[
                int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

        if i > (col_num - 1):  # if not the vn of the first row
            g.append('f_pairwise_vertical' + '{}'.format(
                int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                    x[i])  # edge between each vn and its above-hand fn

    g.compute_marginals()
    for i in range(row_num * col_num):
        # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[j+i*6*16+3*3*6]) / tf.reduce_sum(g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[j+i*6*16+3*3*6], 1 - pi_tilde[j+i*6*16+3*3*6]])
        pi[i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32].assign(
            g.nodes['x' + '{}'.format(i)].marginal()[0])
        # pi[i + 3 * 3*3 * 6 + 5 * 5 * 6 * 16 +3*3*16*32].assign(message_real)

    p01_row = 0.325  # 0.015  ###############for dense###################
    p11_row = 0.458  # 0.085
    p01_col = 0.325  # 0.015
    p11_col = 0.458  # 0.085 all 1

    prob_start = np.array([0.3, 1 - 0.3])  # define start prob #### 0.0005

    for kab in range(8):
        # dense_layer_1   576*120
        row_num = 72  # row number of support matrix
        col_num = 120  # col number of support matrix

        g = FactorGraph(silent=True)  # init the graph without message printouts

        x = []  # list of vn (variable node)
        f_unary = []  # unary factor node
        f_start = []  # unary factor node at the beginning of matrix #####

        prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                            [p01_row, 1 - p01_row]])  # distribution of pairwise horizontal factor node
        f_pairwise_horizontal = []  # horizontal pairwise factor node
        prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                        [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
        f_pairwise_vertical = []  # vertical pairwise factor node
        for i in range(row_num * col_num):
            x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
            # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[i + 5*5*6 + 5*5*6*16] * (1 - pi[i + 5*5*6 + 5*5*6*16]) / (pi[i + 5*5*6 + 5*5*6*16] + (1 - 2 * pi[i + 5*5*6 + 5*5*6*16]) * pi_tilde[i + 5*5*6 + 5*5*6*16]), 1 - pi_tilde[i + 5*5*6 + 5*5*6*16] * (1 - pi[i + 5*5*6 + 5*5*6*16]) / (pi[i + 5*5*6 + 5*5*6*16] + (1 - 2 * pi[i + 5*5*6 + 5*5*6*16]) * pi_tilde[i + 5*5*6 + 5*5*6*16])])))  # define unary factor node
            f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[
                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120] * (
                                                                            pi[
                                                                                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120]) / (
                                                                                    1 - (pi[
                                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120] + (
                                                                                                1 - 2 * pi[
                                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120]) *
                                                                                        pi_tilde[
                                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120])),
                                                                        1 - pi_tilde[
                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120] * (
                                                                            pi[
                                                                                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120]) / (
                                                                                    1 - (
                                                                                    pi[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120] + (
                                                                                            1 - 2 * pi[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120]) *
                                                                                    pi_tilde[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120]))])))  # define unary factor node

            if i in range(row_num * (col_num - 1)):
                f_pairwise_horizontal.append(Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

            if i in range(col_num * (row_num - 1)):
                f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

            if i < col_num or i % col_num == 0:
                f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

        # connect the vn and fns
        for i in range(row_num * col_num):
            g.add(x[i])
            g.add(f_unary[i])
            g.append('f_unary' + '{}'.format(i), x[i])

            if i < col_num:  ####
                g.append('x' + '{}'.format(i), f_start[i])
            elif i % col_num == 0:
                g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

            if (i % col_num) != (col_num - 1):  # if not the vn of the last column
                g.append('x' + '{}'.format(i),
                        f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

            if (i % col_num) != 0:  # if not the vn of the first column
                g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                        x[i])  # edge between each vn and its left-hand fn

            if i < (row_num * col_num - col_num):  # if not the vn of the last row
                g.append('x' + '{}'.format(i), f_pairwise_vertical[
                    int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

            if i > (col_num - 1):  # if not the vn of the first row
                g.append('f_pairwise_vertical' + '{}'.format(
                    int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                        x[i])  # edge between each vn and its above-hand fn

        g.compute_marginals()
        for i in range(row_num * col_num):
            # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[i + 5*5*6 + 5*5*6*16]) / tf.reduce_sum(g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[i + 5*5*6 + 5*5*6*16], 1 - pi_tilde[i + 5*5*6 + 5*5*6*16]])
            pi[
                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + kab * 72 * 120].assign(
                g.nodes['x' + '{}'.format(i)].marginal()[0])
            # pi[j+neuron_num_1*input_size].assign(message_real)

    p01_row = 0.283  # 0.28  # 0.015  ###############for dense###################
    p11_row = 0.485  # 0.485  # 0.085
    p01_col = 0.283  # 0.28  # 0.015
    p11_col = 0.485  # 0.485  # 0.085 all 1

    for kab in range(3):
        # dense_layer_2: 120*84
        row_num = 40  # row number of support matrix
        col_num = 84  # col number of support matrix

        g = FactorGraph(silent=True)  # init the graph without message printouts

        x = []  # list of vn (variable node)
        f_unary = []  # unary factor node
        f_start = []  # unary factor node at the beginning of matrix #####

        prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                            [p01_row, 1 - p01_row]])  # distribution of pairwise horizontal factor node
        f_pairwise_horizontal = []  # horizontal pairwise factor node
        prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                        [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
        f_pairwise_vertical = []  # vertical pairwise factor node
        for i in range(row_num * col_num):
            x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
            # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120] * (1 - pi[i + 5*5*6 + 5*5*6*16 + 400*120]) / (pi[i + 5*5*6 + 5*5*6*16 + 400*120] + (1 - 2 * pi[i + 5*5*6 + 5*5*6*16 + 400*120]) * pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120]), 1 - pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120] * (1 - pi[i + 5*5*6 + 5*5*6*16 + 400*120]) / (pi[i + 5*5*6 + 5*5*6*16 + 400*120] + (1 - 2 * pi[i + 5*5*6 + 5*5*6*16 + 400*120]) * pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120])])))  # define unary factor node
            f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[
                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84] * (
                                                                            pi[
                                                                                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84]) / (
                                                                                    1 - (
                                                                                    pi[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84] + (
                                                                                            1 - 2 * pi[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84]) *
                                                                                    pi_tilde[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84])),
                                                                        1 - pi_tilde[
                                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84] * (
                                                                            pi[
                                                                                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84]) / (
                                                                                    1 - (
                                                                                    pi[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84] + (
                                                                                            1 - 2 * pi[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84]) *
                                                                                    pi_tilde[
                                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84]))])))  # define unary factor node

            if i in range(row_num * (col_num - 1)):
                f_pairwise_horizontal.append(Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

            if i in range(col_num * (row_num - 1)):
                f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

            if i < col_num or i % col_num == 0:
                f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

        # connect the vn and fns
        for i in range(row_num * col_num):
            g.add(x[i])
            g.add(f_unary[i])
            g.append('f_unary' + '{}'.format(i), x[i])

            if i < col_num:  ####
                g.append('x' + '{}'.format(i), f_start[i])
            elif i % col_num == 0:
                g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

            if (i % col_num) != (col_num - 1):  # if not the vn of the last column
                g.append('x' + '{}'.format(i),
                        f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

            if (i % col_num) != 0:  # if not the vn of the first column
                g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                        x[i])  # edge between each vn and its left-hand fn

            if i < (row_num * col_num - col_num):  # if not the vn of the last row
                g.append('x' + '{}'.format(i), f_pairwise_vertical[
                    int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

            if i > (col_num - 1):  # if not the vn of the first row
                g.append('f_pairwise_vertical' + '{}'.format(
                    int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                        x[i])  # edge between each vn and its above-hand fn

        g.compute_marginals()
        for i in range(row_num * col_num):
            # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120]) / tf.reduce_sum(g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120], 1 - pi_tilde[i + 5*5*6 + 5*5*6*16 + 400*120]])
            pi[
                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + kab * 40 * 84].assign(
                g.nodes['x' + '{}'.format(i)].marginal()[0])
            # pi[j+neuron_num_1*input_size+neuron_num_1*neuron_num_2].assign(message_real)

    p01_row = 0.282  # 0.285  # 0.015  ###############for dense###################
    p11_row = 0.48  # 0.485  # 0.085
    p01_col = 0.282  # 0.285  # 0.015
    p11_col = 0.48  # 0.485  # 0.085 all 1

    # dense_layer_3: 84*10
    row_num = 84  # row number of support matrix
    col_num = 10  # col number of support matrix

    g = FactorGraph(silent=True)  # init the graph without message printouts

    x = []  # list of vn (variable node)
    f_unary = []  # unary factor node
    f_start = []  # unary factor node at the beginning of matrix #####

    prob_pairwise_horizontal = np.array([[p11_row, 1 - p11_row],
                                        [p01_row, 1 - p01_row]])  # distribution of pairwise horizontal factor node
    f_pairwise_horizontal = []  # horizontal pairwise factor node
    prob_pairwise_vertical = np.array([[p11_col, 1 - p11_col],
                                    [p01_col, 1 - p01_col]])  # distribution of pairwise vertical factor node
    f_pairwise_vertical = []  # vertical pairwise factor node
    for i in range(row_num * col_num):
        x.append(Variable('x' + '{}'.format(i), 2))  # define 25 vn, each with 2 states
        # f_unary.append(Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84] * (1 - pi[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]) / (pi[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84] + (1 - 2 * pi[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]) * pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]), 1 - pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84] * (1 - pi[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]) / (pi[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84] + (1 - 2 * pi[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]) * pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84])])))  # define unary factor node
        f_unary.append(
            Factor('f_unary' + '{}'.format(i), np.array([pi_tilde[
                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84] * (
                                                            pi[
                                                                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84]) / (
                                                                    1 - (
                                                                    pi[
                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84] + (
                                                                            1 - 2 * pi[
                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84]) *
                                                                    pi_tilde[
                                                                        i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84])),
                                                        1 - pi_tilde[
                                                            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84] * (
                                                            pi[
                                                                i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84]) / (
                                                                1 - (
                                                                pi[
                                                                    i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84] + (
                                                                        1 - 2 * pi[
                                                                    i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84]) *
                                                                pi_tilde[
                                                                    i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84]))])))  # define unary factor node

        if i in range(row_num * (col_num - 1)):
            f_pairwise_horizontal.append(Factor('f_pairwise_horizontal' + '{}'.format(i), prob_pairwise_horizontal))

        if i in range(col_num * (row_num - 1)):
            f_pairwise_vertical.append(Factor('f_pairwise_vertical' + '{}'.format(i), prob_pairwise_vertical))

        if i < col_num or i % col_num == 0:
            f_start.append(Factor('f_start' + '{}'.format(i), prob_start))

    # connect the vn and fns
    for i in range(row_num * col_num):
        g.add(x[i])
        g.add(f_unary[i])
        g.append('f_unary' + '{}'.format(i), x[i])

        if i < col_num:  ####
            g.append('x' + '{}'.format(i), f_start[i])
        elif i % col_num == 0:
            g.append('x' + '{}'.format(i), f_start[col_num + int(i / col_num) - 1])

        if (i % col_num) != (col_num - 1):  # if not the vn of the last column
            g.append('x' + '{}'.format(i),
                    f_pairwise_horizontal[i - int(i / col_num)])  # edge between each vn and its right-hand fn

        if (i % col_num) != 0:  # if not the vn of the first column
            g.append('f_pairwise_horizontal' + '{}'.format(i - int(i / col_num) - 1),
                    x[i])  # edge between each vn and its left-hand fn

        if i < (row_num * col_num - col_num):  # if not the vn of the last row
            g.append('x' + '{}'.format(i), f_pairwise_vertical[
                int(i / col_num) + (row_num - 1) * (i % col_num)])  # edge between each vn and its below-hand fn

        if i > (col_num - 1):  # if not the vn of the first row
            g.append('f_pairwise_vertical' + '{}'.format(
                int((i - col_num) / col_num) + (row_num - 1) * ((i - col_num) % col_num)),
                    x[i])  # edge between each vn and its above-hand fn

    g.compute_marginals()
    for i in range(row_num * col_num):
        # message_real = (g.nodes['x' + '{}'.format(i)].marginal()[0] / pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]) / tf.reduce_sum(
        #     g.nodes['x' + '{}'.format(i)].marginal() / [pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84],
        #                                                 1 - pi_tilde[i +  3 * 3 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3*3*32*64 + 3*3*64*128 + 128*120 + 120*84]])
        pi[
            i + 7 * 7 * 3 * 6 + 5 * 5 * 6 * 16 + 3 * 3 * 16 * 32 + 3 * 3 * 32 * 64 + 576 * 120 + 120 * 84].assign(
            g.nodes['x' + '{}'.format(i)].marginal()[0])
        # pi[j+neuron_num_1*input_size+neuron_num_1*neuron_num_2].assign(message_real)

    # pi.assign(tf.clip_by_value(quantization(pi), 1e-8, 1 - 1e-8))  # quantize pi
    # #####################pi bits##########################
    # bits_pi = 8 * (np.count_nonzero(pi) - np.count_nonzero(pi == 1.0))
    # ######################################################
    if ppv < 10 or ppv % 25 == 0:
        # plt.close('all')
        pi_dense_1 = np.reshape(pi[int(tf.math.reduce_prod(mu_shapes[0]) + tf.math.reduce_prod(
            mu_shapes[1]) + tf.math.reduce_prod(mu_shapes[2]) + tf.math.reduce_prod(mu_shapes[3])):int(
            tf.math.reduce_prod(mu_shapes[0]) + tf.math.reduce_prod(mu_shapes[1]) + tf.math.reduce_prod(
                mu_shapes[2]) + tf.math.reduce_prod(mu_shapes[3]) + tf.math.reduce_prod(mu_shapes[4]))],
                            (576, 120))
        sns_pi_dense_1 = sns.heatmap(pi_dense_1)
        plt.title("pi_dense_1")
        plt.show()
        #############################################
        pi_w2_iter1 = np.reshape(pi[int(
            tf.math.reduce_prod(mu_shapes[0]) + tf.math.reduce_prod(mu_shapes[1]) + tf.math.reduce_prod(
                mu_shapes[2]) + tf.math.reduce_prod(mu_shapes[3]) + tf.math.reduce_prod(mu_shapes[4])):int(
            tf.math.reduce_prod(mu_shapes[0]) + tf.math.reduce_prod(mu_shapes[1]) + tf.math.reduce_prod(
                mu_shapes[2]) + tf.math.reduce_prod(mu_shapes[3]) + tf.math.reduce_prod(mu_shapes[4]) + tf.math.reduce_prod(
                mu_shapes[5]))],
                                (120, 84))
        sns_pi_w2 = sns.heatmap(pi_w2_iter1)
        plt.title("pi_dense_2")
        plt.show()
        #####################################################
        pi_test_3 = np.reshape(pi[int(tf.math.reduce_prod(mu_shapes[0]) + tf.math.reduce_prod(
            mu_shapes[1]) + tf.math.reduce_prod(mu_shapes[2]) + tf.math.reduce_prod(mu_shapes[3]) + tf.math.reduce_prod(
            mu_shapes[4]) + tf.math.reduce_prod(mu_shapes[5])):int(
            tf.math.reduce_prod(mu_shapes[0]) + tf.math.reduce_prod(mu_shapes[1]) + tf.math.reduce_prod(
                mu_shapes[2]) + tf.math.reduce_prod(mu_shapes[3]) + tf.math.reduce_prod(mu_shapes[4]) + tf.math.reduce_prod(
                mu_shapes[5]) + tf.math.reduce_prod(mu_shapes[6]))],
                            (84, 10))
        sns_pi_dense_3 = sns.heatmap(pi_test_3)
        plt.title("pi_dense_3")
        plt.show()
    #######################################################
    ################################Module A############################################################################
    a_tilde_error = 1
    b_tilde_error = 1
    pi_tilde_error = 1
    u = 1
    ############################################################s-step#####################################################
    # s�ֲ�����
    temp1.assign(pi * (b ** a) / tf.math.exp(tf.math.lgamma(a)) * tf.math.exp(
        (a - 1.0) * (tf.math.digamma(a_tilde) - tf.math.log(b_tilde)) - b * (
                a_tilde / b_tilde)))
    temp1.assign(tf.clip_by_value(temp1, 1e-10, 1e1000000))
    temp2.assign(
        (1.0 - pi) * (b_bar ** a_bar) / tf.math.exp(tf.math.lgamma(a_bar)) * tf.math.exp(
            (a_bar - 1.0) * (tf.math.digamma(a_tilde) - tf.math.log(b_tilde)) - b_bar * (
                    a_tilde / b_tilde)))
    pi_tilde.assign(temp1 / (temp1 + temp2))

    pi_tilde_kuozhan.assign(pi_tilde)  # ��pi_tilde_kuozhan��ֵ

    pi_tilde_error = tf.norm(pi_tilde - pi_tilde_old) / tf.norm(pi_tilde)
    print('pi_tilde_error=', pi_tilde_error)
    if np.all(pi_tilde - pi_tilde_old) < 1e-6:
        pi_tilde_error_local = 1e-6  # ############################################################## ����nan��������������
    pi_tilde_old.assign(pi_tilde)
    #####################################################################################################
    ############################################################rho-step##################################################
    # rho�ֲ�����
    a_tilde.assign(pi_tilde_kuozhan * a + (1 - pi_tilde_kuozhan) * a_bar + np.ones(len(trainable_variable_mu)))
    b_tilde.assign(tf.cast(
        tf.square(trainable_variable_mu) + trainable_variable_var + tf.cast(pi_tilde_kuozhan * b,
                                                                            dtype=tf.float32) + tf.cast(
            (1 - pi_tilde_kuozhan) * b_bar, dtype=tf.float32), dtype=tf.double))
    new_prior_rho.assign(a_tilde / b_tilde)
    new_prior_sigma.assign(1.0 / np.sqrt(new_prior_rho))  # need to be feedback to clients
    print('new_prior_sigma=',tf.norm(new_prior_sigma))
    a_tilde_error = tf.norm(tf.abs(a_tilde - a_tilde_old)) / tf.norm(tf.abs(a_tilde))
    b_tilde_error = tf.norm(tf.abs(b_tilde - b_tilde_old)) / tf.norm(tf.abs(b_tilde))
    if np.all(a_tilde - a_tilde_old) < 0.008:
        a_tilde_error = 1e-6
    if np.all(b_tilde - b_tilde_old) < 0.008:
        b_tilde_error = 1e-6
    a_tilde_old.assign(a_tilde)
    b_tilde_old.assign(b_tilde)
    ###################################################w-step########################################################
    weights_mu_local_list = []  # store weights_mu from each client
    weights_rho_local_list = []  # store weights_rho from each client
    bias_b_local_list = []  # store bias_b from each client

    avg_local_sparsity = 0
    temp_sum_sparsity = 0
    up_load_bits = 0
    for i in range(num_clients):  # each client perform w-step
        accuracy, weights_mu_local, weights_rho_local, bias_b_local = local_train(l_rate, batch_size,
                                                                                  local_dataset_size[i],
                                                                                  local_train_features_list[i],
                                                                                  local_train_labels_list[i],
                                                                                  aggregated_weights_mu,
                                                                                  aggregated_weights_rho,
                                                                                  aggregated_bias, test_features,
                                                                                  test_labels,
                                                                                  new_prior_sigma)
        if ppv < 30:
            prune_threshold = 0.01
        # elif ppv < 30 & ppv >= 10:
        #     prune_threshold = 0.01
        else:
            prune_threshold = 0.01
        weights_mu_local_pruned = local_prunning_mag(weights_mu_local, prune_threshold, prune_threshold)
        weights_mu_local_quantized = quantization_list(weights_mu_local_pruned)
        weights_mu_local_list.append(weights_mu_local_pruned)
        weights_rho_local_list.append(weights_rho_local)
        bias_b_local_list.append(bias_b_local)
        print('local_accuracy_this_client,', accuracy)

        bits_this_client = count_bits_list(weights_mu_local_pruned)
        up_load_bits = up_load_bits + bits_this_client

        sparsity_this_client = calculate_sparsity_list(weights_mu_local_pruned)
        temp_sum_sparsity = temp_sum_sparsity + sparsity_this_client

    avg_local_sparsity = temp_sum_sparsity / num_clients
    avg_local_sparsity_list.append(avg_local_sparsity)

    up_load_bits_list.append(up_load_bits)

    ## aggregation###
    aggregated_weights_mu = aggregation_list(weights_mu_local_list, num_clients, local_dataset_size)  # aggregated weights_mu
    aggregated_weights_var = aggregation_rho_list(weights_rho_local_list,
                                                  num_clients, local_dataset_size)  # aggregated weights_variance
    aggregated_weights_rho = [0 for i in range(len(aggregated_weights_var))]  # construct aggregated rho list
    for i in range(len(aggregated_weights_var)):
        assert len(aggregated_weights_var) == len(
            aggregated_weights_rho)  # assert aggregated var length == aggregated rho list
        aggregated_weights_rho[i] = tf.Variable(tfp.math.softplus_inverse(tf.sqrt(aggregated_weights_var[i])),
                                                name='weights_rho')  # calculate aggregateed rho according to aggregated variance; need to be feedback to clients

    aggregated_bias = aggregation_list(bias_b_local_list, num_clients, local_dataset_size)  # aggregated bias

    aggregated_weights_mu_flat = [0 for i in range(len(aggregated_weights_mu))]  # ����һ����list�����������ƽ���
    for i in range(len(aggregated_weights_mu)):
        aggregated_weights_mu_flat[i] = tf.reshape(aggregated_weights_mu[i], [-1])

    aggregated_weights_var_flat = [0 for i in range(len(aggregated_weights_var))]  # ����һ����list�����������ƽ���
    for i in range(len(aggregated_weights_var)):
        aggregated_weights_var_flat[i] = tf.reshape(aggregated_weights_var[i], [-1])

    trainable_variable_mu = tf.concat(
        [aggregated_weights_mu_flat[0], aggregated_weights_mu_flat[1], aggregated_weights_mu_flat[2],
         aggregated_weights_mu_flat[3], aggregated_weights_mu_flat[4], aggregated_weights_mu_flat[5], aggregated_weights_var_flat[6]], 0)

    trainable_variable_var = tf.concat(
        [aggregated_weights_var_flat[0], aggregated_weights_var_flat[1], aggregated_weights_var_flat[2],
         aggregated_weights_var_flat[3], aggregated_weights_var_flat[4], aggregated_weights_var_flat[5], aggregated_weights_var_flat[6]], 0)

    ###################count bits##########################
    down_load_bits = count_bits_list(aggregated_weights_mu)/2*3  # mu,var,prior_var
    down_load_bits_list.append(down_load_bits)
    total_bits = total_bits + down_load_bits + up_load_bits
    total_bits_list.append(total_bits)
    ###############calculate sparsity######################
    sparse_threshold = 0.01
    num_elements_gt = 0  # total number of nonzero elements in aggregated model
    for i in range(len(aggregated_weights_mu)):
        elements_gt = tf.math.greater_equal(abs(aggregated_weights_mu[i]), sparse_threshold)
        num_elements_gt = num_elements_gt + tf.math.reduce_sum(tf.cast(elements_gt, tf.int32))

    sparsity_this_round = num_elements_gt / total_weight_number
    sparsity_per_round.append(sparsity_this_round)
    print("sparsity_this_round==", sparsity_this_round)
    #######################################global test accuracy########################################################
    accuracy_sum = 0
    for ib in range(100):  # divide test set into 100 batches
        test_predictions = forward_run_deter(test_features[ib * 100: (ib + 1) * 100], aggregated_weights_mu,
                                             aggregated_bias)  # �õ�test����Predictions

        pred_error = tf.keras.losses.sparse_categorical_crossentropy(test_labels[ib * 100: (ib + 1) * 100],
                                                                     test_predictions, from_logits=True)  # �õ�pred_error
        # print('*****pred_error:', pred_error)
        prediction_probability = tf.nn.softmax(test_predictions)
        test_predictions_labels = tf.argmax(prediction_probability, axis=1)
        accuracy = tf.reduce_sum(
            tf.cast(tf.equal(test_predictions_labels, test_labels[ib * 100: (ib + 1) * 100]),
                    test_predictions_labels.dtype)) / len(test_labels[ib * 100: (ib + 1) * 100])
        accuracy_sum = accuracy_sum + accuracy

    accuracy_avg = accuracy_sum / 100
    print('Global_test_acc_this_round:', accuracy_avg)
    accuracy_global.append(accuracy_avg)
    if ppv < 10 or ppv % 25 == 0:
        plt.plot(accuracy_global)
        plt.show()
    ########################plot figures###############
    if ppv < 10 or ppv % 25 == 0:
        pi_tilde_dense_1 = np.reshape(pi_tilde[int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])):int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])+tf.math.reduce_prod(mu_shapes[5]))],
                                    (120, 84))

        prior_var = b_tilde / a_tilde
        prior_var_dense_1 = np.reshape(prior_var[int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])):int(tf.math.reduce_prod(mu_shapes[0])+tf.math.reduce_prod(mu_shapes[1])+tf.math.reduce_prod(mu_shapes[2])+tf.math.reduce_prod(mu_shapes[3])+tf.math.reduce_prod(mu_shapes[4])+tf.math.reduce_prod(mu_shapes[5]))],
                                    (120, 84))

        aggregated_pi_tilde_conv1 = tf.reshape(pi_tilde[:882], [7, 7, 3, 6])
        pi_conv1 = tf.reshape(pi[:882], [7, 7, 3, 6])
        # np.savetxt('pi_tilde_w2_fl_mnist_iter1.csv', pi_tilde_w2_iter1, delimiter=',')
        aggregated_w_mu_dense_1 = aggregated_weights_mu[5]
        aggregated_w_mu_conv1 = aggregated_weights_mu[0]
        aggregated_w_var_dense_1 = aggregated_weights_var[5]

        sns_aggregated_pi_tilde_dense_1 = sns.heatmap(pi_tilde_dense_1)
        plt.title("pi_tilde_dense_1")
        plt.show()

        sns_aggregated_w_mu_dense1 = sns.heatmap(tf.clip_by_value(abs(aggregated_w_mu_dense_1), 0, 100))
        plt.title("aggregated_w_mu_dense_1")
        plt.show()

        sns_aggregated_w_var_dense1 = sns.heatmap(tf.clip_by_value(abs(aggregated_w_var_dense_1), 0, 100))
        plt.title("aggregated_w_var_dense_1")
        plt.show()

        sns_prior_var_dense_1 = sns.heatmap(prior_var_dense_1)
        plt.title("prior_var_dense_1")
        plt.show()

        fig, ax = plt.subplots(3, 6)  # ÿ�зֱ�conv_1��weights_mu, pi, jbjb
        for i in range(6):
            sns.heatmap(tf.abs(aggregated_w_mu_conv1[:, :, 0, i]), ax=ax[0][i])
            sns.heatmap(pi_conv1[:, :, 0, i], ax=ax[1][i])

        plt.show()
    ###########################plot figures end#################################
    print('a_tilde_error', a_tilde_error)
    print('b_tilde_error', b_tilde_error)
    print('pi_tilde_error', pi_tilde_error)
    print(pi_tilde)
    a_tilde_error_list.append(a_tilde_error)
    b_tilde_error_list.append(b_tilde_error)
    pi_tilde_error_list.append(pi_tilde_error)

    u = u + 1
    ######end of Module A##################
    # l_rate = l_rate * 0.93  #0.93 good 1203
    # l_rate = l_rate * 0.9
    # if ppv % 10 == 0:
    #     l_rate = l_rate * 0.9
    # if l_rate < 1e-4:
    #     l_rate = 1e-4
    ppv = ppv + 1
    print('current_round:', ppv)
    # l_rate = l_rate / (2*ppv)


#########################################################################
for i in range(len(aggregated_weights_mu)):
    plt.hist(np.array(tf.abs(tf.reshape(aggregated_weights_mu[i], [-1]))), bins=100, range=(0,5))
    plt.show()
#######################################################################
# prune_threshold = 0.1
# prune_threshold_list = [0.5, 0.5, 0.1, 0.1, 0.1]
# weights_mu_aggregated_pruned = local_prunning_mag(aggregated_weights_mu, prune_threshold_list)
# #############################################################################
# weights_mu_dense2 = weights_mu_aggregated_pruned[3]
# weights_mu_conv1 = weights_mu_aggregated_pruned[0]
# weights_mu_conv2 = weights_mu_aggregated_pruned[1]
# sns_weights_mu_dense2 = sns.heatmap(tf.clip_by_value(abs(weights_mu_dense2),0,3))
# plt.title("weights_mu_dense2")
# plt.show()
# fig, ax = plt.subplots(3, 6)  # ÿ�зֱ�conv_1��weights_mu, pi, jbjb
# for i in range(6):
#     sns.heatmap(tf.abs(weights_mu_conv1[:, :, 0, i]), ax=ax[0][i])
#     sns.heatmap(pi_conv1[:, :, 0, i], ax=ax[1][i])
# plt.show()
# ###############################################  global model test
# test_predictions_global = forward_run_deter(test_features, weights_mu_aggregated_pruned, aggregated_bias)  # �õ�test����Predictions
#
# pred_error = tf.keras.losses.sparse_categorical_crossentropy(test_labels,
#                                                              test_predictions_global)  # �õ�pred_error
# print('*****pred_error:', pred_error)
# test_predictions_labels = tf.argmax(test_predictions_global, axis=1)
# accuracy = tf.reduce_sum(
#     tf.cast(tf.equal(test_predictions_labels, test_labels), test_predictions_labels.dtype)) / 10000
# print('*******pruned_ACCURACY:', accuracy)
# ################################################pruned sparsity
# sparse_threshold = 0.01
# num_elements_gt = 0  # total number of nonzero elements in aggregated model
# for i in range(len(weights_mu_aggregated_pruned)):
#     elements_gt = tf.math.greater_equal(abs(weights_mu_aggregated_pruned[i]), sparse_threshold)
#     num_elements_gt = num_elements_gt + tf.math.reduce_sum(tf.cast(elements_gt, tf.int32))
#
# sparsity_this_round = num_elements_gt / total_weight_number
# sparsity_per_round.append(sparsity_this_round)
# print("sparsity_this_round==", sparsity_this_round)
# #############################################################
# pruned_kernel = 0
# for i in range(6):
#         if tf.reduce_sum(tf.abs(weights_mu_conv1[:,:,0,i])) <= 10**-3:
#             pruned_kernel = pruned_kernel+1
#
# print("conv1_pruned_kernel=", pruned_kernel)
#
# pruned_kernel = 0
# for i in range(6):
#     for j in range(16):
#         if tf.reduce_sum(tf.abs(weights_mu_conv2[:,:,i,j])) <= 10**-3:
#             pruned_kernel = pruned_kernel+1
#             print(i,j)
#
# print("conv1_pruned_kernel=", pruned_kernel)
# for i in range(2,5):
#     print(np.sum(np.sum(np.sum(np.abs(weights_mu_aggregated_pruned[i]), axis=1) > 10 ** -3)))
#     print(np.where(np.sum(np.abs(weights_mu_aggregated_pruned[i]), axis=1) < 10 ** -3))
#
# ##################################FINE TUNING###########################################
# epoch_num = 10
# weights_mu_ft, bias_b_ft, mask_ft, accuracy_ft = fine_tune(l_rate, epoch_num, batch_size_1, train_size_1, train_features_1, train_labels_1, weights_mu_aggregated_pruned, bias_b_aggregated, test_features, test_labels)
# #######################################################################################
#
# def create_adversarial_pattern(input_image, input_label):  # generate adversaral pattern
#     with tf.GradientTape() as tape:
#         tape.watch(input_image)
#         prediction = forward_run_deter(input_image, weights_mu_ft, bias_b_ft)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction, from_logits=True)
#     gradient = tape.gradient(loss, input_image)
#     # ���ݶ�ʹ��sign�����������Ŷ�
#     signed_grad = tf.sign(gradient)
#     return signed_grad
#
# perturbations = create_adversarial_pattern(test_features, test_labels)   # generate pertubation pattern
# plt.imshow(perturbations[0])  # plot pertuebation pattern
#
# def display_images(image, label,adv_label,num=10):  # image:original image, label: orginal label, adv_label, num: # of plotted figures every time
#     fig = plt.figure(figsize=(2*num,3)) # figsize:ָ��figure�Ŀ��͸ߣ���λΪӢ��
#     for i in range(num):   # pre_image��shape�ĵ�һ��ά�Ⱦ��Ǹ�����������num
#         plt.subplot(1,num,i+1) # ���м��е� ��i+1��ͼƬ����1��ʼ��
#         plt.imshow((image[i,:,:,:] + 1)/2) # ��1��2: �����ɵ�-1��1��ͼƬŪ��0-1֮��,
#         plt.title('{} -> {}'.format(label[i],adv_label[i]))
#         plt.axis('off') # ��Ҫ����
#     plt.show()
#
# # �ڲ�ͬ��epsilons���в���
# epsilons = [0,0.05,0.10,0.15,0.20,0.25,0.30]
# adv_acc_list = []
# for i, eps in enumerate(epsilons):
#   print("epsilons = {}:".format(eps))
#   prediction = forward_run_deter(test_features, weights_mu_ft, bias_b_ft)  # get predict labels
#   predict_label = np.argmax(prediction, axis=1)
#   # generate adv examples
#   adv_features = test_features + eps * perturbations
#   adv_features = tf.clip_by_value(adv_features, 0, 1)
#   adv_prediction = forward_run_deter(adv_features, weights_mu_ft, bias_b_ft)  # get adv predict labels
#   adv_predict_label = np.argmax(adv_prediction, axis=1)
#   adv_test_accuracy = np.sum(test_labels == adv_predict_label) / len(test_labels)
#   adv_acc_list.append(adv_test_accuracy)
#   # ��ͼ
#   display_images(adv_features, predict_label, adv_predict_label)
#
#
#
# adv_features_train = test_features + 0.1 * perturbations
# new_train_features_adv = tf.concat([train_features_1[:50000], adv_features_train],0)
# new_train_labels_adv = np.concatenate((train_labels_1[:50000],test_labels), 0)


