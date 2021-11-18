#https://github.com/MingtaoGuo/DCGAN_WGAN_WGAN-GP_LSGAN_SNGAN_RSGAN_BEGAN_ACGAN_PGGAN_TensorFlow/blob/master/GANs.py

import tensorflow as tf
import scipy.io as sio
import numpy as np
import math
import os
import sys 
import matplotlib.pyplot as plt
import time 
import matplotlib.gridspec as gridspec
import preprocess_for_gan
import argparse
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='GAN')

    # basic parameters
    parser.add_argument('--phase', type=str, default='train', help='train or generate')
    parser.add_argument('--GAN_type', type=str, default='WGAN-GP', help='the name of the model: WGAN, WGAN-GP, DCGAN, LSGAN, SNGAN, RSGAN, RaSGAN')
    parser.add_argument('--data_dir', type=str, default= 'data\\0HP', help='the directory of the data')
    parser.add_argument('--target', type=str, default='B007', help='target signal to generate')
    parser.add_argument('--imbalance_ratio', type=int, default=100, help='imbalance ratio between major class samples and minor class samples')
    parser.add_argument('--checkpoint_dir', type=str, default='samples\WGAN-GP\ORDER\ratio-50\orderIR021-05-17-20_22\checkpoint\model-97000', help='the saved checkpoint to generate signals')
    parser.add_argument('--batch_size', type=int, default=5, help='batchsize of the training process')
    parser.add_argument('--swith_threshold', type=int, default=2.5, help='threshold of G-D loss difference for determining to train G or D')
    parser.add_argument('--normalization', type=str, default='minmax', help='way to process data: minmax or mean')
    parser.add_argument('--sampling', type=str, default='order', help='way to sample signals from original dataset: enc, order, random')


    # optimization information
    parser.add_argument('--lr', type=float, default=2e-4, help='the initial learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-14, help='if epsilon is too big, training of DCGAN is failure')
   

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=30000, help='max number of epoch')
    parser.add_argument('--sample_step', type=int, default=1000, help='the interval of log training information')

    args = parser.parse_args()
    return args


def get_sin_training_data(shape):
    '''
    使用预加载的数据_正弦信号
    '''
    half_T = 30 # T/2 of sin function
    length = shape[0] * shape[1]
    array = np.arange(0, length)
    ori_data = np.sin(array*np.pi/half_T)
    training_data = np.reshape(ori_data, shape)
    return training_data

def shuffle_set(x, y):
    size = np.shape(x)[0]
    x_row = np.arange(0, size)
    permutation = np.random.permutation(x_row.shape[0])
    x_shuffle = x[permutation,:,:]
    y_shuffle = np.array(y)[permutation]
    return x_shuffle, y_shuffle
    
def get_batch(x, y, now_batch, batch_size, total_batch):
    if now_batch < total_batch - 1:
        x_batch = x[now_batch*batch_size:(now_batch+1)*batch_size,:]
        y_batch = y[now_batch*batch_size:(now_batch+1)*batch_size]
    else:
        x_batch = x[now_batch*batch_size:,:]
        y_batch = y[now_batch*batch_size:]
    return x_batch, y_batch

def plot(samples):
    num = np.size(samples,0)
    length = np.size(samples,1)
    fig = plt.figure()
    gs = gridspec.GridSpec(num,1)
    gs.update(wspace = 0.05, hspace = 0.05)
    samples = np.reshape(samples, [num,length])
    x = np.arange(0, length)
    for i in range(num):
        ax = plt.subplot(gs[i,0])
        y = samples[i]
        ax.plot(x, y)
    return fig

def deconv(inputs, shape, strides, out_num, is_sn=False):
    # input [X_batch, in_channels, in_width] // 2D [X_batch, height, width, in_channels]
    # shape [filter_width, output_channels, in_channels]
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-2]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv1d_transpose(inputs, spectral_norm("sn", filters), out_num, strides) + bias
    else:
        return tf.nn.conv1d_transpose(inputs, filters, out_num, strides, "SAME") + bias

def conv(inputs, shape, strides, is_sn=False):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-1]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv1d(inputs, spectral_norm("sn", filters), strides, "SAME") + bias
    else:
        return tf.nn.conv1d(inputs, filters, strides, "SAME") + bias

def fully_connected(inputs, num_out, is_sn=False):
    W = tf.get_variable("W", [inputs.shape[-1], num_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.matmul(inputs, spectral_norm("sn", W)) + b
    else:
        return tf.matmul(inputs, W) + b

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope*inputs, inputs)

def spectral_norm(name, w, iteration=1):
    #Spectral normalization which was published on ICLR2018,please refer to "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
    #This function spectral_norm is forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def mapping(x, epsilon):
    max = np.max(x)
    min = np.min(x)
    return (x - min) * 255.0 / (max - min + epsilon)

def instanceNorm(inputs, epsilon):
    mean, var = tf.nn.moments(inputs, axes=[1], keep_dims=True) # axes=[1,2]
    scale = tf.get_variable("scale", shape=mean.shape[-1], initializer=tf.constant_initializer([1.0]))
    shift = tf.get_variable("shift", shape=mean.shape[-1], initializer=tf.constant_initializer([0.0]))
    return (inputs - mean) * scale / (tf.sqrt(var + epsilon)) + shift


def make_dir(args):
    
    now = time.strftime("%m-%d-%H_%M", time.localtime(time.time()))

    # samples_dir 
    # if args.phase == 'train':
    samples_dir = "samples/"+args.GAN_type+'/'+'ratio-'+str(args.imbalance_ratio)+'/'+args.sampling+args.target+'-'+now
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    figure_dir = samples_dir + "/figure"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    checkpoint_dir = samples_dir + "/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    signals_dir = samples_dir + "/signals"
    if not os.path.exists(signals_dir):
        os.makedirs(signals_dir)

    return samples_dir, figure_dir, checkpoint_dir, signals_dir

class Generator:
    def __init__(self, name, epsilon):
        self.name = name
        self.epsilon = epsilon

    def __call__(self, Z):
        size = tf.shape(Z)[0]
        with tf.variable_scope(name_or_scope=self.name, reuse=False):
            with tf.variable_scope(name_or_scope="linear"):
                inputs = tf.reshape(tf.nn.relu((fully_connected(Z, 7*256))), [size, 7, 256]) #[random_size]-->[7,256]
            with tf.variable_scope(name_or_scope="deconv1"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 128, 256], [1,2,1],[size, 14, 128]), self.epsilon)) #[7,256]-->[14,128]
            with tf.variable_scope(name_or_scope="deconv2"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 64, 128], [1,4,1], [size, 56, 64]), self.epsilon)) #[14,128]-->[56,64]
            with tf.variable_scope(name_or_scope="deconv3"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 32, 64], [1,4,1], [size, 224, 32]), self.epsilon)) #[56,64]-->[224,32]
            with tf.variable_scope(name_or_scope="deconv4"):
                inputs = tf.nn.tanh(deconv(inputs, [5, 1, 32], [1,4,1], [size, data_dim, 1])) # [224,32]-->[896,1]
                # inputs = deconv(inputs, [5, 1, 32], [1,4,1], [size, data_dim, 1]) # [224,32]-->[896,1]
            return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name, epsilon):
        self.name = name
        self.epsilon = epsilon

    def __call__(self, inputs, gn_stddev, reuse=False, is_sn=False):
        gaussian_nosie = tf.random_normal(shape=tf.shape(inputs), mean=0., stddev=gn_stddev, dtype=tf.float32)
        inputs = inputs + gaussian_nosie
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
            with tf.variable_scope("conv1"):
                inputs = leaky_relu(conv(inputs, [5, 1, 32], [1,4,1], is_sn)) # [896,1]-->[224,32]
            with tf.variable_scope("conv2"):
                inputs = leaky_relu(instanceNorm(conv(inputs, [5, 32, 64], [1,4,1], is_sn), self.epsilon)) #[224,32]-->[56,64]
            with tf.variable_scope("conv3"):
                inputs = leaky_relu(instanceNorm(conv(inputs, [5, 64, 128], [1,4,1], is_sn), self.epsilon)) #[56,64]-->[14,128]
            with tf.variable_scope("conv4"):
                inputs = leaky_relu(instanceNorm(conv(inputs, [5, 128, 256], [1,2,1], is_sn), self.epsilon)) #[14,128]-->[7,256]
            inputs = tf.layers.flatten(inputs)
            return fully_connected(inputs, 1, is_sn) #[7,256]-->[1,1]

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class GAN:
    #Architecture of generator and discriminator just like DCGAN.
    def __init__(self, args, x_train, y_train, number):
        self.args = args
        self.random_dim = 512 #100
        self.epsilon = args.epsilon
        self.number = number
        self.batch_size = args.batch_size
        self.sample_step = args.sample_step
        self.Z = tf.placeholder("float", [None, self.random_dim])
        self.X = tf.placeholder("float", [None, data_dim, 1])
        self.gn_stddev = tf.placeholder("float", [])
        self.x_train = x_train
        self.y_train = y_train
        D = Discriminator("discriminator", self.epsilon)
        G = Generator("generator", self.epsilon)
        self.fake_X = G(self.Z)
        if args.GAN_type == "DCGAN":
            #DCGAN, paper: UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
            self.fake_logit = tf.nn.sigmoid(D(self.fake_X, self.gn_stddev))
            self.real_logit = tf.nn.sigmoid(D(self.X, self.gn_stddev, reuse=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + self.epsilon)) + tf.reduce_mean(tf.log(1 - self.fake_logit + self.epsilon)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + self.epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif args.GAN_type == "WGAN":
            #WGAN, paper: Wasserstein GAN
            self.fake_logit = D(self.fake_X, self.gn_stddev)
            self.real_logit = D(self.X, self.gn_stddev, reuse=True)
            self.d_loss = -tf.reduce_mean(self.real_logit) + tf.reduce_mean(self.fake_logit)
            self.g_loss = -tf.reduce_mean(self.fake_logit)
            self.clip = []
            for _, var in enumerate(D.var):
                self.clip.append(var.assign(tf.clip_by_value(var, -0.01, 0.01)))
            self.opt_D = tf.train.RMSPropOptimizer(5e-5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.RMSPropOptimizer(5e-5).minimize(self.g_loss, var_list=G.var)
        elif args.GAN_type == "WGAN-GP":
            #WGAN-GP, paper: Improved Training of Wasserstein GANs
            self.fake_logit = D(self.fake_X, self.gn_stddev)
            self.real_logit = D(self.X, self.gn_stddev, reuse=True)
            # 1. WGAN_GP
            # e = tf.random_uniform([self.batch_size, 1, 1], 0, 1)
            # x_hat = e * self.X + (1 - e) * self.fake_X
            # grad = tf.gradients(D(x_hat, self.gn_stddev, reuse=True), x_hat)[0]
            # self.d_loss = tf.reduce_mean(self.fake_logit - self.real_logit) + 10 * tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1)) - 1)) #axis=[1,2,3]
            
            # 2. WGAN_div1
            # real_grad = tf.gradients(D(self.X, self.gn_stddev, reuse=True), self.X)[0]
            # fake_grad = tf.gradients(D(self.fake_X, self.gn_stddev, reuse=True), self.fake_X)[0]
            # real_grad_norm = tf.pow(tf.reduce_sum(tf.square(real_grad),axis=[1,2]), 3)
            # fake_grad_norm = tf.pow(tf.reduce_sum(tf.square(fake_grad),axis=[1,2]), 3)
            # grad_pen = tf.reduce_mean(real_grad_norm+fake_grad_norm)
            # self.d_loss = tf.reduce_mean(self.fake_logit - self.real_logit) + grad_pen
            
            # 3. WGAN_div2
            e = tf.random_uniform([self.batch_size, 1, 1], 0, 1)
            x_hat = e * self.X + (1 - e) * self.fake_X
            grad = tf.gradients(D(x_hat, self.gn_stddev, reuse=True), x_hat)[0]
            self.d_loss = tf.reduce_mean(self.fake_logit - self.real_logit) + 2 * tf.reduce_mean(tf.reduce_sum(tf.square(grad), axis=[1,2])) 
            
            self.g_loss = tf.reduce_mean(-self.fake_logit)
            self.opt_D = tf.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(self.g_loss, var_list=G.var)
        elif args.GAN_type == "LSGAN":
            #LSGAN, paper: Least Squares Generative Adversarial Networks
            self.fake_logit = D(self.fake_X, self.gn_stddev)
            self.real_logit = D(self.X, self.gn_stddev, reuse=True)
            self.d_loss = tf.reduce_mean(0.5 * tf.square(self.real_logit - 1) + 0.5 * tf.square(self.fake_logit))
            self.g_loss = tf.reduce_mean(0.5 * tf.square(self.fake_logit - 1))
            self.opt_D = tf.train.AdamOptimizer(5e-5, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(5e-5, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif args.GAN_type == "SNGAN":
            #SNGAN, paper: SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS
            self.fake_logit = tf.nn.sigmoid(D(self.fake_X, self.gn_stddev, is_sn=True))
            self.real_logit = tf.nn.sigmoid(D(self.X, self.gn_stddev, reuse=True, is_sn=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + self.epsilon) + tf.log(1 - self.fake_logit + self.epsilon)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + self.epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif args.GAN_type == "RSGAN":
            #RSGAN, paper: The relativistic discriminator: a key element missing from standard GAN
            self.fake_logit = D(self.fake_X, self.gn_stddev)
            self.real_logit = D(self.X, self.gn_stddev, reuse=True)
            self.d_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(self.real_logit - self.fake_logit) + self.epsilon))
            self.g_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(self.fake_logit - self.real_logit) + self.epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif args.GAN_type == "RaSGAN":
            #RaSGAN, paper: The relativistic discriminator: a key element missing from standard GAN
            self.fake_logit = D(self.fake_X, self.gn_stddev)
            self.real_logit = D(self.X, self.gn_stddev, reuse=True)
            self.avg_fake_logit = tf.reduce_mean(self.fake_logit)
            self.avg_real_logit = tf.reduce_mean(self.real_logit)
            self.D_r_tilde = tf.nn.sigmoid(self.real_logit - self.avg_fake_logit)
            self.D_f_tilde = tf.nn.sigmoid(self.fake_logit - self.avg_real_logit)
            self.d_loss = - tf.reduce_mean(tf.log(self.D_r_tilde + self.epsilon)) - tf.reduce_mean(tf.log(1 - self.D_f_tilde + self.epsilon))
            self.g_loss = - tf.reduce_mean(tf.log(self.D_f_tilde + self.epsilon)) - tf.reduce_mean(tf.log(1 - self.D_r_tilde + self.epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.G_saver = tf.train.Saver(G.var)

    def __call__(self):
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
        # saver = tf.train.Saver(max_to_keep=5)
        args = self.args
        gn_stddev = 0.1
        train_G = False
        train_D = False
        iter = math.ceil(self.number/self.batch_size)# 每轮epoch中batch的迭代数
        train_times = 5
        if args.phase == 'train':
            samples_dir, figure_dir, checkpoint_dir, signals_dir = make_dir(args)
            for epoch in range(args.max_epoch):
                x_shuffle, y_shuffle= shuffle_set(self.x_train, self.y_train)
                if epoch % 1000 ==0 and epoch != 0:
                    step = epoch // 1000
                    gn_stddev /= step # stddev decreased every step
                for i in range(iter):
                    Z_batch = np.random.standard_normal([self.batch_size, self.random_dim])
                    # X_batch = get_sin_training_data([self.batch_size, data_dim, 1])
                    X_batch, _= get_batch(x_shuffle, y_shuffle, i, self.batch_size, iter)
                    d_loss = self.sess.run(self.d_loss, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                    g_loss = self.sess.run(self.g_loss, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                    self.sess.run(self.opt_D, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                    if args.GAN_type == "WGAN":
                        self.sess.run(self.clip)#WGAN weight clipping
                    self.sess.run(self.opt_G, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                
                # train G or D if the loss difference between them is too large
                loss_difference = abs(g_loss) - abs(d_loss)
                if loss_difference > args.swith_threshold:
                    train_G = True
                elif loss_difference < - args.swith_threshold:
                    train_D = True
                else:
                    train_D = False
                    train_G = False

                if train_G: #train G for 5 times if train_G
                    for t in range(train_times):
                        for i in range(iter):
                            Z_batch = np.random.standard_normal([self.batch_size, self.random_dim])
                            # X_batch = get_sin_training_data([self.batch_size, data_dim, 1])
                            X_batch, _= get_batch(x_shuffle, y_shuffle, i, self.batch_size, iter)
                            d_loss = self.sess.run(self.d_loss, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                            g_loss = self.sess.run(self.g_loss, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                            self.sess.run(self.opt_G, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                
                if train_D:
                    for t in range(train_times):
                        for i in range(iter):
                            Z_batch = np.random.standard_normal([self.batch_size, self.random_dim])
                            # X_batch = get_sin_training_data([self.batch_size, data_dim, 1])
                            X_batch, _= get_batch(x_shuffle, y_shuffle, i, self.batch_size, iter)
                            d_loss = self.sess.run(self.d_loss, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                            g_loss = self.sess.run(self.g_loss, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})                    
                            self.sess.run(self.opt_D, feed_dict={self.X: X_batch, self.Z: Z_batch, self.gn_stddev: gn_stddev})
                            if args.GAN_type == "WGAN":
                                self.sess.run(self.clip)#WGAN weight clipping

                if epoch % 100 == 0:
                    # print("epoch: %d, step: %d, d_loss: %g, g_loss: %g"%(epoch, i,  d_loss, g_loss))
                    print("epoch: {}, d_loss: {}, g_loss: {}".format(epoch, d_loss, g_loss))
                    sys.stdout.flush()

                if epoch % self.sample_step == 0:
                    z = np.random.standard_normal([50, self.random_dim]) 
                    samples = self.sess.run(self.fake_X, feed_dict={self.Z: z})
                    sample_piece = samples[0:1,:]
                    fig = plot(sample_piece)
                    fig.savefig('{}/{}.png'.format(figure_dir, str(epoch//self.sample_step).zfill(4)), bbox_inches = 'tight')
                    plt.close(fig)

                    sio.savemat('{}/{}.mat'.format(signals_dir, str(epoch//self.sample_step).zfill(4), bbox_inches = 'tight'),{'x1':samples})
                    saver.save(self.sess, checkpoint_dir+"/model", global_step = epoch)
        
        else:
            
            if args.target == 'B007':
                file_name = '12k_Drive_End_B007_0_118'
            elif args.target == 'B014':
                file_name = '12k_Drive_End_B014_0_185'
            elif args.target == 'B021':
                file_name = '12k_Drive_End_B021_0_222'
            elif args.target == 'IR007':
                file_name = '12k_Drive_End_IR007_0_105'
            elif args.target == 'IR014':
                file_name = '12k_Drive_End_IR014_0_169'
            elif args.target == 'IR021':
                file_name = '12k_Drive_End_IR021_0_209'
            elif args.target == 'OR007':
                file_name = '12k_Drive_End_OR007@6_0_130'
            elif args.target == 'OR014':
                file_name = '12k_Drive_End_OR014@6_0_197'
            elif args.target == 'OR021':
                file_name = '12k_Drive_End_OR021@6_0_234'
            
            # G_saver = tf.train.Saver(G.var)
            # self.G_saver.restore(self.sess, tf.train.latest_checkpoint("samples/WGAN-GP/ORDER/ratio-50/orderOR021-05-05-00_27/checkpoint"))
            self.G_saver.restore(self.sess, args.checkpoint_dir)
            z = np.random.standard_normal([1000, self.random_dim]) 
            samples = self.sess.run(self.fake_X, feed_dict={self.Z: z})
            sio.savemat('{}/{}.mat'.format("generated_data/ORDER_minmax_ratio50", file_name),{'DE':samples})




if __name__ == "__main__":
    args = parse_args()

    data_dim = 896 # 大于2周期
    diagnosis_number = 1000
    rate = [0.5,0.25,0.25] # 测试集验证集划分比例
    number = int(diagnosis_number*rate[0]//args.imbalance_ratio) # 训练样本的数量, 等于故障诊断模型的训练样本量

    x_train, y_train = preprocess_for_gan.prepro(d_path=args.data_dir,
                                            target=args.target,
                                            length=data_dim,
                                            number=number,
                                            normalization=args.normalization,
                                            rate=rate,
                                            sampling=args.sampling,
                                            imbalance_ratio = args.imbalance_ratio
                                            )
    # 输入卷积的时候还需要修改一下，增加通道数目
    x_train = x_train[:,:,np.newaxis]
    gan = GAN(args, x_train, y_train, number)
    gan()