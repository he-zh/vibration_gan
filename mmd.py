import torch
from torch.autograd import Variable
import preprocess_for_gan
import os
from scipy.io import loadmat
import numpy as np
import preprocess_for_diagnosis

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

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
    # 增加'0-1'归一化处理
        generated_data[i,:] = (generated_data[i,:]- min(generated_data[i,:]))/(max(generated_data[i,:])-min(generated_data[i,:]))
    generated_data = np.asarray(generated_data).reshape(-1,896)
    return generated_data



if __name__ == "__main__":
    # 训练参数
    path = r'data/0HP'
    g_path = r'generated_data\ORDER_minmax_ratio50'
    # target_1 = 'OR021'
    length = 896 # 大于2周期
    normalization='0-1' #'minmax', 'mean'
    list= ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']

    imbalance_ratio = 100
    over_sampling = 'GAN' #'GAN', 'SMOTE', 'ADASYN','RANDOM', 'sampling_method'. 默认'none'
    number = 1000
    generated = 'generated_data/ORDER_minmax_ratio50'
    rate = [0.5,0.25,0.25] # 测试集验证集划分比例
    num = int(number*rate[0])
    minor_num = num//imbalance_ratio
    sampling = 'order'
    compare_num = 200         
    # x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_for_diagnosis.prepro(d_path=path,
    #                                                                                     gan_data=generated,
    #                                                                                     length=length,
    #                                                                                     number=number,
    #                                                                                     normalization=normalization,
    #                                                                                     rate=rate,
    #                                                                                     sampling=sampling, 
    #                                                                                     over_sampling=over_sampling,
    #                                                                                     imbalance_ratio=imbalance_ratio)  
    start = 9*minor_num+num
    g_num = num-minor_num
                                                                                                                                                                   
    for i in range(9):
        target_1 = list[i]
        target_2 = list[i]
        real_data, _ = preprocess_for_gan.prepro(d_path=path,
                                                    target=target_1,
                                                    length=length,
                                                    number=compare_num,
                                                    normalization=normalization,
                                                    rate=[0.5,0.25,0.25],
                                                    sampling='random',
                                                    imbalance_ratio = 1
                                                    )
        #与自己比较
        # generated_data, _ = preprocess_for_gan.prepro(d_path=path,
        #                                         target=target_2,
        #                                         length=data_dim,
        #                                         number=num,
        #                                         normalization=normalization,
        #                                         rate=[0.8,0.1,0.1],
        #                                         sampling='random',
        #                                         imbalance_ratio = 1
        #                                         )
        
        #与GAN生成的数据比较
        generated_data = load_gdata(g_path=g_path,target=target_2,number=compare_num)

        #与OVER SAMPLING比较
        # generated_data = x_train[start:start+compare_num,:]
        # start += g_num

        X = torch.Tensor(real_data)
        Y = torch.Tensor(generated_data)
        X,Y = Variable(X), Variable(Y)
        print('{} : {}'.format(target_1,mmd_rbf(X,Y)))