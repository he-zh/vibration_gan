import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import os
from sklearn import preprocessing 
from numpy import random
import preprocess_for_gan
from scipy.fftpack import fft

def capture(path,filenames):
    files = {}
    for i in filenames:
        # 文件路径
        file_path = os.path.join(path, i)
        file = loadmat(file_path)
        file_keys = file.keys()
        for key in file_keys:
            if 'DE' in key:
                files[i] = file[key].ravel()
    return files

def slice(data):
    # keys = data.keys()
    # sample = {}
    keys = data.keys()

    Samples = {}
    for i in keys:
        sample = []
        slice_data = data[i]
        # random_start = np.random.randint(low=0, high=(len(slice_data)))
        random_start = 0
        sample = slice_data[random_start:random_start+100000]
        # Samples.append(sample)
        sample = np.asarray(sample).reshape([1,-1])
    return sample

def load_gdata(g_path, target, number):
    filenames = os.listdir(g_path)
    # gan模式只从小类(故障)文件中读取数据
    filename = [i for i in filenames if (target in i) and ('.mat' in i)]
    file_path = os.path.join(g_path, filename[0])
    file = loadmat(file_path)
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:
            generated_data = file[key]
    generated_data = generated_data[0:number,0:896]
    generated_data = np.asarray(generated_data).reshape(-1,896)
    return generated_data

def normilization(sample):
    # 用训练集标准差标准化训练集以及测试集
    sample[0,:] = 2*(sample[0,:]- min(sample[0,:]))/(max(sample[0,:])-min(sample[0,:]))-1
    # sample[0,:] = sample[0,:]- np.mean(sample[0,:])/(max(sample[0,:])-min(sample[0,:]))
    # sample[0,:] = (sample[0,:]-np.mean(sample[0,:]))/np.std(sample[0,:])
   
    return sample

def plot(real_samples,generated_samples):
    length = 896
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(9,2)
    gs.update(wspace = 0.05, hspace = 0.05)
    list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']
    # samples = np.reshape(samples, [num,length])
    # x = np.arange(0, length//2+1).reshape([length//2+1])
    # x = 12000/896*x
    x = np.arange(0, length).reshape([length])
    x = x/12000
    for i in range(9):
        ax = plt.subplot(gs[i,0])
        y_r = real_samples[i].reshape([length])
        ax.plot(x, y_r,'b',lw=1)
        # fft_r=abs(fft(y_r)).reshape([length])
        # fft_r = fft_r[0:length//2+1]/(length//2)
        # ax.plot(x, fft_r,'b',lw=1)
        if i !=8 : ax.set_xticklabels([])        
        # plt.ylim(0,0.16)
        # plt.yticks([0,0.1])

        # ax.plot(x, y_r,'b',lw=0.5)
        bx = plt.subplot(gs[i,1])
        y_g = generated_samples[i].reshape([length])
        bx.plot(x, y_g,'b',lw=1)
        # fft_g=abs(fft(y_g)).reshape([length])
        # fft_g = fft_g[0:length//2+1]/(length//2)
        # bx.plot(x, fft_g,'b',lw=1)
        if i !=8 : bx.set_xticklabels([])  
        # plt.ylim(0,0.16)
        # plt.yticks([0,0.1])
        # bx.plot(x, y_g,'b',lw=0.5)
        # ax.scatter(x,y)

    plt.show()
    return fig

# path = "data/0HP"
# target = 'OR021'
# # 获得该文件夹下所有.mat文件名
# filenames = os.listdir(path)
# filename = [i for i in filenames if (target in i) and ('.mat' in i)]
# data = capture(path, filename)
# sample = slice(data)
# sample = normilization(sample)
# fig = plot(sample)
# fig.savefig('{}/{}.png'.format('data/0HP/data8000', target+'-minmax-8000'), bbox_inches = 'tight')
# # plt.close(fig)

path = r'data/0HP'
g_path = r'generated_data\bestlooking'
# target_1 = 'OR021'
length = 896 # 大于2周期
normalization='minmax' #'minmax', 'mean'
list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']
over_sampling = 'GAN' #'GAN', 'SMOTE', 'ADASYN','RANDOM', 'sampling_method'. 默认'none'
number = 1
generated = 'generated_data/ORDER_minmax_ratio50'
rate = [0.5,0.25,0.25] # 测试集验证集划分比例
sampling = 'order'
real_samples = []
generated_samples = []
for i in range(9):
    target_1 = list[i]
    target_2 = list[i]
    real_data, _ = preprocess_for_gan.prepro(d_path=path,
                                                target=target_1,
                                                length=length,
                                                number=number,
                                                normalization=normalization,
                                                rate=[0.5,0.25,0.25],
                                                sampling='order',
                                                imbalance_ratio = 1
                                                )
    real_samples.append(real_data)                                       
    generated_data = load_gdata(g_path=g_path,target=target_2,number=number)
    generated_samples.append(generated_data)
fig = plot(real_samples,generated_samples)