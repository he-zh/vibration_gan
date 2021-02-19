from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import preprocess_for_diagnosis
from keras.callbacks import TensorBoard
import numpy as np
import time 
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='diagnosis')

    # basic parameters
    parser.add_argument('--data_dir', type=str, default= 'data\\0HP', help='the directory of the data')
    parser.add_argument('--generated_data_dir', type=str, default=None, help='directory of generated data if oversampling_method=GAN')
    parser.add_argument('--imbalance_ratio', type=int, default=100, help='imbalance ratio between major class samples and minor class samples')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--normalization', type=str, default='minmax', help='way to process data: minmax or mean')
    parser.add_argument('--sampling', type=str, default='order', help='way to sample signals from original dataset: enc, order, random')
    parser.add_argument('--oversampling_method', type=str, default='none', help='way to oversample data:GAN, SMOTE, ADASYN,RANDOM, sampling_method')

    # switch between train and test phase
    parser.add_argument('--phase', type=str, default='train', help='to train the diagnosis model or test it')
   
    # save, load information
    parser.add_argument('--max_epoch', type=int, default=600, help='max number of epoch')
    parser.add_argument('--checkpoint_dir', type=str, default='diagnosis_CWRU_model/ADASYN/order-50-0.71-best.hdf5', help='the saved checkpoint of diagnosis model')

    args = parser.parse_args()
    return args

args = parse_args()

# 训练参数
length = 896
number = 1000 # 每类样本的数量/大类样本的数量
rate = [0.5,0.25,0.25] # 测试集验证集划分比例
BatchNorm = True
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=args.data_dir,
                                                                                    gan_data=args.generated_data_dir,
                                                                                    length=length,
                                                                                    number=number,
                                                                                    normalization=args.normalization,
                                                                                    rate=rate,
                                                                                    sampling=args.sampling, 
                                                                                    over_sampling=args.oversampling_method,
                                                                                    imbalance_ratio=args.imbalance_ratio,
                                                                                    )                                                                                   
num_classes = y_test.shape[1]
# 输入卷积的时候还需要修改一下，增加通道数目
x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]
# 输入数据的维度
input_shape =x_train.shape[1:]
print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

now = time.strftime("%d-%H_%M", time.localtime(time.time()))

logdir = 'logs/'+args.oversampling_method+'/'+ args.sampling+ str(args.imbalance_ratio) +'/'+now
save_path = 'diagnosis_CWRU_model/' + args.oversampling_method

if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 定义卷积层
def wdcnn(filters, kernerl_size, strides, conv_padding, pool_padding,  pool_size, BatchNormal):
    """wdcnn层神经元

    :param filters: 卷积核的数目，整数
    :param kernerl_size: 卷积核的尺寸，整数
    :param strides: 步长，整数
    :param conv_padding: 'same','valid'
    :param pool_padding: 'same','valid'
    :param pool_size: 池化层核尺寸，整数
    :param BatchNormal: 是否Batchnormal，布尔值
    :return: model
    """
    model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
                     padding=conv_padding, kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))
    return model

# 实例化序贯模型
model = Sequential()
# 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来
model.add(Conv1D(filters=16, kernel_size=64, strides=7, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

# 第二层卷积

model = wdcnn(filters=32, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid',  pool_size=2, BatchNormal=BatchNorm) # [64,16]-->[32,32]
# 第三层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm) # [32,32]-->[16,64]
# 第四层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm) # [16,64]-->[8,64]
# 第五层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm) #[8,64]-->[4,64]
# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
# 增加输出层
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))



if args.phase == 'train':
    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

    # TensorBoard调用查看一下训练情况
    tb_cb = TensorBoard(log_dir=logdir)
    #checkpoint
    filepath = save_path+'/'+ args.sampling+ '-'+str(args.imbalance_ratio)+'-{val_accuracy:.2f}-best.hdf5'
    # 开始模型训练
    checkpoint= ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    callbacks_list= [tb_cb, checkpoint]
    model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.max_epoch,
            verbose=2, validation_data=(x_valid, y_valid), shuffle=True,
            callbacks=callbacks_list)
else:
    model.load_weights(args.checkpoint_dir)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

# 评估模型
score = model.evaluate(x=x_train, y=y_train, verbose=0)
print("测试集上的损失：", score[0])
print("测试集上的准确率:",score[1])
# plot_model(model=model, to_file='wdcnn.png', show_shapes=True)
# model_name = "10_minmax_order_balance.h5"

# model.save(save_path+'best_val_acc.h5')
# model.save('diagnosis_CWRU_model/multi_classi_normal_imbalanced_10.h5')