import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import preprocess_for_gan
import os
from scipy.io import loadmat
import numpy as np
from scipy.fftpack import fft,ifft

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
    for i in range(np.size(generated_data,0)):
        generated_data = np.asarray(generated_data).reshape(-1,896)
    return generated_data

def fft_transform(data):
    dim = np.size(data,0)
    fft_data = []
    for i in range(dim):           
        fft_datapiece=abs(fft(data[i,:])) 
        fft_data.append(fft_datapiece)
    fft_data=np.asarray(fft_data).reshape(dim,-1)
    return fft_data


path = r'data/0HP'
g_path = r'generated_data\ORDER_minmax_ratio10'
data_dim = 896 # 大于2周期
diagnosis_number = 1000
normalization='minmax' #'minmax', 'mean'



list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021','normal']
# list = ['B007','normal']
X = []
for i in range(len(list)):
    target = list[i]
    real_data, _ = preprocess_for_gan.prepro(d_path=path,
                                                target=target,
                                                length=data_dim,
                                                number=50,
                                                normalization=normalization,
                                                rate=[0.5,0.25,0.25],
                                                sampling='random',
                                                imbalance_ratio = 1
                                                )
    if target != 'normal':
        generated_data = load_gdata(g_path=g_path,target=target,number=number)
        temp_X = np.concatenate((real_data,generated_data),axis=0)
    else: 
        temp_X = real_data
    if i == 0: X = temp_X
    else: X = np.concatenate((X,temp_X),axis=0)
X = fft_transform(X)


'''t-SNE'''
tsne = manifold.TSNE(n_iter=1000,n_components=2,perplexity=10, init='pca')
X_tsne = tsne.fit_transform(X)
# print("Org data dimension is {}. Embedded data dimension is {}".format(real_data.shape[-1], X_tsne.shape[-1]))
'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})

# for i in range(len(list)):
#     plt.scatter(X_norm[i*number:(i+1)*number,0],X_norm[i*number:(i+1)*number,1],color=plt.cm.Set1(i),label='r_'+list[i],alpha=0.8, s=20)
#     plt.legend(loc='best', framealpha=0)
for i in range(len(list)):
    k = 2*i
    plt.scatter(X_norm[k*number:(k+1)*number,0],X_norm[k*number:(k+1)*number,1],color=plt.cm.tab20(k),label='r_'+list[i],s=15)
    plt.scatter(X_norm[(k+1)*number:(k+2)*number,0],X_norm[(k+1)*number:(k+2)*number,1],edgecolors=plt.cm.tab20(k+1),marker='o',c='',label='g_'+list[i], s=15)
    # plt.legend(loc='best', framealpha=0)
    plt.legend(bbox_to_anchor=(1.001, 0), loc=3, borderaxespad=0)
plt.show()