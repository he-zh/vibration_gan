from scipy.io import loadmat
import numpy as np
import os 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import preprocess_for_diagnosis

def capture(path):
    file = loadmat(path)
    return file

def slice(data):
    x_test = data['x1']
    x_test = x_test[:,0:896]
    # for i in range(np.size(x_test,0)):
        # 增加[-1,1]归一化处理
        # x_test[i,:] = 2*(x_test[i,:]- min(x_test[i,:]))/(max(x_test[i,:])-min(x_test[i,:]))-1
        # x_test[i,:] = (x_test[i,:]- np.mean(x_test[i,:]))/(max(x_test[i,:])-min(x_test[i,:]))
    return x_test

def load_gdata(g_path, target):
    filenames = os.listdir(g_path)
    # gan模式只从小类(故障)文件中读取数据
    filename = [i for i in filenames if (target in i) and ('.mat' in i)]
    file_path = os.path.join(g_path, filename[0])
    file = loadmat(file_path)
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:
            generated_data = file[key]
    generated_data = generated_data[:,0:896]
    generated_data = np.asarray(generated_data).reshape(-1,896)
    return generated_data
model = load_model('diagnosis_CWRU_model/none-order-1-best.h5')


# 测试原始数据训练集的准确率
# imbalance_ratio = 100
# over_sampling = 'none' #'GAN', 'SMOTE', 'ADASYN','RANDOM', 'sampling_method'. 默认'none'
# path = r'data/0HP'
# length = 896 # 大于2周期
# normalization='minmax' #'minmax', 'mean'
# number = 1000
# generated = 'generated_data/ORDER_minmax_ratio10'
# rate = [0.5,0.25,0.25] # 测试集验证集划分比例
# sampling = 'order'
# num = int(number*rate[0])//imbalance_ratio
# list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']
# x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=path,
#                                                                                     gan_data=generated,
#                                                                                     length=length,
#                                                                                     number=number,
#                                                                                     normalization=normalization,
#                                                                                     rate=rate,
#                                                                                     sampling=sampling, 
#                                                                                     over_sampling=over_sampling,
#                                                                                     imbalance_ratio=imbalance_ratio,
#                                                                                     )      
# x_train = x_train[:,:,np.newaxis]      
# start = 0  
# for i in range(9):
#     x_trainpiece = x_train[start:start+num,:,:]
#     y_trainpiece = y_train[start:start+num,:]
#     start += num
#     score = model.evaluate(x=x_trainpiece, y=y_trainpiece, verbose=0)
#     print("{} : {}".format(list[i],score[1]))



#测试某个.mat文件准确率

# path = r"samples\WGAN-GP\ORDER\ratio-50\orderIR021-05-17-20_22\signals\0061.mat"
# data = capture(path)
# x_train = slice(data)
# num = x_train.shape[0]
# y_train = np.ones((num,10), dtype=int)
# # y_train = [1,0,0,0,0,0,0,0,0,0]*y_train
# # y_train = [0,1,0,0,0,0,0,0,0,0]*y_train
# # y_train = [0,0,1,0,0,0,0,0,0,0]*y_train
# # y_train = [0,0,0,1,0,0,0,0,0,0]*y_train
# # y_train = [0,0,0,0,1,0,0,0,0,0]*y_train
# y_train = [0,0,0,0,0,1,0,0,0,0]*y_train
# # y_train = [0,0,0,0,0,0,1,0,0,0]*y_train
# # y_train = [0,0,0,0,0,0,0,1,0,0]*y_train
# # y_train = [0,0,0,0,0,0,0,0,1,0]*y_train
# score = model.evaluate(x=x_train, y=y_train, verbose=0)
# # print("测试集上的损失：", score[0])
# print(score[1])


# 展示9类GAN生成样本的分类准确率

y_label = [[1,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,1,0]]
list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']
path = 'generated_data/ORDER_minmax_ratio50'
for i in range(len(list)):
    target = list[i]
    x_train = load_gdata(g_path=path,target=target)
    x_train = x_train[:,:,np.newaxis]
    num = x_train.shape[0]
    y_train = np.ones((num,10), dtype=int)
    y_train = y_label[i]*y_train   
    # print('label', model.predict(x_train))
    score = model.evaluate(x=x_train, y=y_train, verbose=0)
    # print("测试集上的损失：", score[0])
    print("{} : {}".format(target,score[1]))



#展示imlearn生成样本的分类准确率
# imbalance_ratio = 10
# over_sampling = 'RANDOM' #'GAN', 'SMOTE', 'ADASYN','RANDOM', 'sampling_method'. 默认'none'
# path = r'data/0HP'
# length = 896 # 大于2周期
# normalization='minmax' #'minmax', 'mean'
# number = 1000
# generated = 'generated_data/ORDER_minmax_ratio10'
# rate = [0.5,0.25,0.25] # 测试集验证集划分比例
# num = int(number*rate[0])
# minor_num = num//imbalance_ratio
# sampling = 'order'
# display_n = 100
# list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']
# x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=path,
#                                                                                     gan_data=generated,
#                                                                                     length=length,
#                                                                                     number=number,
#                                                                                     normalization=normalization,
#                                                                                     rate=rate,
#                                                                                     sampling=sampling, 
#                                                                                     over_sampling=over_sampling,
#                                                                                     imbalance_ratio=imbalance_ratio)                                                                                          
# x_train = x_train[:,:,np.newaxis]                                                                                   
# start = 9*minor_num+num
# g_num = num-minor_num
# for i in range(9):
#     x_trainpiece = x_train[start:start+g_num,:,:]
#     y_trainpiece = y_train[start:start+g_num,:]
#     start += g_num
#     score = model.evaluate(x=x_trainpiece, y=y_trainpiece, verbose=0)
#     print("{} : {}".format(list[i],score[1]))



#计算训练好的CNN的测试准确率

# model.load_model(r'diagnosis_CWRU_model\ADASYN\order-50-0.71-best.hdf5')
# classification = '10' #'binary' '10'
# if classification == 'binary':
#     path = r'data/0HP Binary Classification' # 10分类
# else:
#     path = r'data/0HP'
# generated = 'generated_data/ORDER_minmax_ratio10'
# length = 896
# number = 1000 # 每类样本的数量/大类样本的数量
# normalization='minmax' # 最大最小值归一化'minmax', 均值归一化'mean'
# rate = [0.5,0.25,0.25] # 测试集验证集划分比例
# sampling = 'order'
# over_sampling = 'none' #'GAN', 'SMOTE', 'ADASYN','RANDOM', 'sampling_method'. 默认'none'
# imbalance_ratio = 1
# batch_size = 256
# epochs = 600
# BatchNorm = True
# x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=path,
#                                                                                     gan_data=generated,
#                                                                                     length=length,
#                                                                                     number=number,
#                                                                                     normalization=normalization,
#                                                                                     rate=rate,
#                                                                                     sampling=sampling, 
#                                                                                     over_sampling=over_sampling,
#                                                                                     imbalance_ratio=imbalance_ratio,
#                                                                                     )    
# x_train, x_test = x_train[:,:,np.newaxis], x_test[:,:,np.newaxis] 
# score = model.evaluate(x=x_test, y=y_test, verbose=0)
# print(score[0])