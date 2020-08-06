import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import preprocess_for_gan
import os
from scipy.io import loadmat
import numpy as np
from scipy.fftpack import fft,ifft
import preprocess_for_diagnosis
def fft_transform(data):
    dim = np.size(data,0)
    fft_data = []
    for i in range(dim):           
        fft_datapiece=abs(fft(data[i,:])) 
        fft_data.append(fft_datapiece)
    fft_data=np.asarray(fft_data).reshape(dim,-1)
    return fft_data

number = 1000
imbalance_ratio = 50
rate = [0.5,0.25,0.25] # 测试集验证集划分比例
num = int(number*rate[0])
minor_num = num//imbalance_ratio
display_r = 50
display_g = 50
real_number = display_r - minor_num #真实数据(不包含不平衡训练集)
over_sampling = 'SMOTE' #'GAN', 'SMOTE', 'ADASYN','RANDOM', 'sampling_method'. 默认'none'
path = r'data/0HP'
length = 896 # 大于2周期
normalization='minmax' #'minmax', 'mean'

generated = 'generated_data/ORDER_minmax_ratio'+str(imbalance_ratio)

sampling = 'order'
list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021','normal']
generated_data, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=path,
                                                                                    gan_data=generated,
                                                                                    length=length,
                                                                                    number=number,
                                                                                    normalization=normalization,
                                                                                    rate=rate,
                                                                                    sampling=sampling, 
                                                                                    over_sampling=over_sampling,
                                                                                    imbalance_ratio=imbalance_ratio,
                                                                                    )        
real_data, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=path,
                                                                                    gan_data=generated,
                                                                                    length=length,
                                                                                    number=2*real_number, #display_r
                                                                                    normalization=normalization,
                                                                                    rate=rate,
                                                                                    sampling='order', 
                                                                                    over_sampling='none',
                                                                                    imbalance_ratio=1,
                                                                                    )    
# gdata_samples = []
# rdata_samples = []
gdata_samples = np.empty([0,896])
rdata_samples = np.empty([0,896])
ibdata_samples = np.empty([0,896])
if over_sampling == 'GAN':
    start = minor_num #int(number*rate[0])//imbalance_ratio, 5 or 10
    g_start = 0
    r_start = 0
    for i in range(9):
        # gdata_samples.append(generated_data[start:start+display_g,:])
        gdata_samples = np.concatenate((gdata_samples,generated_data[start:start+display_g,:]),axis=0)
        # rdata_samples.append(generated_data[g_start:g_start+minor_num,:]) #不平衡数据集
        ibdata_samples = np.concatenate((ibdata_samples,generated_data[g_start:g_start+minor_num,:]),axis=0)
        # rdata_samples.append(real_data[r_start:r_start+real_number,:]) #平衡 数据集
        rdata_samples = np.concatenate((rdata_samples,real_data[r_start:r_start+real_number,:]),axis=0)
        start += num
        g_start += num
        r_start += real_number
    # rdata_samples.append(generated_data[g_start:g_start+display_r,:])#正常数据
    rdata_samples = np.concatenate((rdata_samples,generated_data[g_start:g_start+display_r,:]),axis=0)
else:
    start = 9*minor_num+num
    g_num = num-minor_num # 450 or 490
    g_start = 0
    r_start = 0
    for i in range(9):
        # gdata_samples.append(generated_data[start:start+display_g,:])
        gdata_samples = np.concatenate((gdata_samples,generated_data[start:start+display_g,:]),axis=0)
        # rdata_samples.append(generated_data[g_start:g_start+minor_num,:]) #不平衡数据集
        ibdata_samples = np.concatenate((ibdata_samples,generated_data[g_start:g_start+minor_num,:]),axis=0)
        # rdata_samples.append(real_data[r_start:r_start+real_number,:]) #平衡 数据集
        rdata_samples = np.concatenate((rdata_samples,real_data[r_start:r_start+real_number,:]),axis=0)
        start += g_num
        g_start += minor_num
        r_start += real_number
    # rdata_samples.append(generated_data[g_start:g_start+display_r,:])
    rdata_samples = np.concatenate((rdata_samples,generated_data[g_start:g_start+display_r,:]),axis=0)

# start = 0
# for i in range(10):
#     rdata_samples.append(real_data[start:start+minor_num,:])
#     start += minor_num
# rdata_samples = np.asarray(rdata_samples).flatten().reshape(-1,896)
# gdata_samples = np.asarray(gdata_samples).reshape(-1,896)
X = np.concatenate((rdata_samples,gdata_samples),axis=0)
X = np.concatenate((X,ibdata_samples),axis=0)
X = fft_transform(X)


'''t-SNE'''
tsne = manifold.TSNE(n_iter=1000,n_components=2,perplexity=50, init='pca',random_state=1) #n_iter=1000,perplexity=8
X_tsne = tsne.fit_transform(X)
# print("Org data dimension is {}. Embedded data dimension is {}".format(real_data.shape[-1], X_tsne.shape[-1]))
'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
fig=plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})

# for i in range(len(list)):
#     plt.scatter(X_norm[i*number:(i+1)*number,0],X_norm[i*number:(i+1)*number,1],color=plt.cm.Set1(i),edgecolors=plt.cm.tab20(2*i+1),label='r_'+list[i],alpha=0.8, s=20)
#     plt.legend(loc='best', framealpha=0)


r_start = 0
g_start = 9*real_number + display_r
ib_start= g_start+9*display_g
for i in range(len(list)):   
    # plt.scatter(X_norm[r_start:r_start+500//imbalance_ratio,0],X_norm[r_start:r_start+500//imbalance_ratio,1],color=plt.cm.tab20(2*i),label='r_'+list[i],alpha=0.8,s=10)
    plt.scatter(X_norm[r_start:r_start+real_number,0],X_norm[r_start:r_start+real_number,1],color=plt.cm.tab20(2*i),label='real '+list[i],s=30)
    if list[i]!='normal': plt.scatter(X_norm[g_start:g_start+display_g,0],X_norm[g_start:g_start+display_g,1],edgecolors=plt.cm.tab20(2*i+1),marker='o',c='',label='generated '+list[i], s=30)
    if list[i]!='normal': plt.scatter(X_norm[ib_start:ib_start+minor_num,0],X_norm[ib_start:ib_start+minor_num,1],color=plt.cm.tab20(2*i),marker='x',label='imbalanced '+list[i], s=30)
    r_start += real_number
    g_start += display_g
    ib_start+= minor_num
    # plt.legend(loc='best', framealpha=0)
    # plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
plt.title('t-sne of {}-generated data at imbalanced ratio {}'.format(over_sampling,imbalance_ratio))
plt.show()
f_path = 'T-SNE/balanceimbalance_realdata/X_maker'
if not os.path.exists(f_path):
    os.makedirs(f_path)
fig.savefig('{}/{}-{}-B021-p50'.format(f_path,over_sampling,imbalance_ratio),dpi=300)