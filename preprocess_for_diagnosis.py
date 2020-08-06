from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import random
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


def prepro(d_path, gan_data=None, length=896, number=1000, normalization='minmax', rate=[0.5, 0.25, 0.25], sampling = 'order', over_sampling = 'none', imbalance_ratio = 10):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param gan_data: 是否采用gan生成的数据作为数据源, 默认为none, 若over_sampling='GAN', 应为存放gan数据的文件夹
    :param length: 信号长度，默认约2个信号周期，896
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normalization: 数据变换方式: 最大最小值归一化'minmax', 均值归一化'mean', 归一化为0-1之间'0-1'
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param sampling: 训练数据的采样方式. 顺序采样'order', 随机采样'random', 重叠采样'enc'
    :param imbalance_ratio: 某一种大类数据与小类数据的比例,默认为10:1
    :param over_sampling: 是否扩增小类数据,可选方案为'GAN', 'SMOTE', 'ADASYN', 'RANDOM', 'sampling_method'. 默认'none'
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path,
                                                                    gan_data=None, 
                                                                    length=896, 
                                                                    number=1000, 
                                                                    normalization='minmax', 
                                                                    rate=[0.5, 0.25, 0.25], 
                                                                    sampling = 'order', 
                                                                    over_sampling = False, 
                                                                    imbalance_ratio = 10
                                                                    )
    ```
    """


    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(original_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice_sampling(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例, 并采样.

        :param data: 单条数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        majority_key = 'normal' # 大类样本关键字
        ratio = imbalance_ratio # 大类数据和小类数据的不平衡比
        for key in keys:
            slice_data = data[key]
            all_lenght = len(slice_data)
            if majority_key in key:
                end_index = int(all_lenght * (1 - slice_rate))
                samp_train = int(number * (1 - slice_rate))
            elif over_sampling == 'sampling_method':
                end_index = int(all_lenght * (1 - slice_rate) // ratio + 2*length)
                samp_train = int(number * (1 - slice_rate))
            else:
                end_index = int(all_lenght * (1 - slice_rate) // ratio+ 2*length)
                samp_train = int(number * (1 - slice_rate) // ratio)
            Train_sample = []
            Test_sample = []
            if sampling == 'enc':
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数 
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            elif sampling == 'random':
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)
            elif sampling == 'order':
                samp_step = 0  # 用来计数Train采样次数 
                step_num = 0
                step = max(int((end_index - length) // samp_train), 1)
                for j in range(samp_train):               
                    order_start = int(step_num * step)
                    step_num += 1
                    if order_start > length:
                        order_start = 0
                        step_num = 0
                    time = (end_index - order_start) // length
                    label = 0                 
                    for h in range(time):
                        samp_step += 1
                        sample = slice_data[order_start:order_start + length]
                        order_start += length
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break  
            else:
                print("please make sampling = 'enc', 'random' or 'order'")

            # 抓取测试数据
            for h in range(number - int(number * (1 - slice_rate))):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_sample.append(sample)
            
            if normalization == 'minmax':
                for i in range(np.size(Train_sample,0)):
                    Train_sample[i] = 2*(Train_sample[i]- min(Train_sample[i]))/(max(Train_sample[i])-min(Train_sample[i]))-1
                for i in range(np.size(Test_sample,0)):
                    Test_sample[i] = 2*(Test_sample[i]- min(Test_sample[i]))/(max(Test_sample[i])-min(Test_sample[i]))-1
            elif normalization == 'mean':
                for i in range(np.size(Train_sample,0)):
                    Train_sample[i] = (Train_sample[i]- np.mean(Train_sample[i]))/(max(Train_sample[i])-min(Train_sample[i]))
                for i in range(np.size(Test_sample,0)):
                    Test_sample[i] = (Test_sample[i]- np.mean(Test_sample[i]))/(max(Test_sample[i])-min(Test_sample[i]))
            elif normalization == '0-1':
                for i in range(np.size(Train_sample,0)):
                    Train_sample[i] = (Train_sample[i]- min(Train_sample[i]))/(max(Train_sample[i])-min(Train_sample[i]))
                for i in range(np.size(Test_sample,0)):
                    Test_sample[i] = (Test_sample[i]- min(Test_sample[i]))/(max(Test_sample[i])-min(Test_sample[i]))
           
            Train_Samples[key] = Train_sample
            Test_Samples[key] = Test_sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        
        return Train_Y, Test_Y

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test
    def pre_gandata(gan_data, d_num):
        """
        处理生成的数据, 加标签
        """
        minor_filenames = os.listdir(gan_data)
        G_data = {}
        for i in minor_filenames:
            # 文件路径
            file_path = os.path.join(gan_data, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    slice_file = file[key]
                    leng = list(range(slice_file.shape[0]))
                    random.shuffle(leng)
                    index = leng[0:d_num]
                    G_data[i] = slice_file[index,0:896].reshape([-1,896])
        return G_data

    def data_standardization(Train_X, Test_X): 
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    enc_step=28
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    filenames = [i for i in filenames if '.mat' in i]
    # 从.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_sampling(data)
    # if normalization:
    #     Train_X, Test_X = data_standardization(Train_X, Test_X)
    if over_sampling == 'GAN':
        if gan_data:
            smaple_num = int(number*rate[0]*(1-1/imbalance_ratio))
            G_data = pre_gandata(gan_data, smaple_num)
            file_keys = G_data.keys()
            for k in file_keys:
                for n in range(G_data[k].shape[0]):
                    array_G = G_data[k]
                    train[k].append(array_G[n,:])
   
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    if over_sampling == 'SMOTE':
        oversample = SMOTE(k_neighbors=2) 
        Train_X, Train_Y = oversample.fit_resample(Train_X, Train_Y)
    elif over_sampling == 'ADASYN':
        oversample = ADASYN(n_neighbors=3) 
        Train_X, Train_Y = oversample.fit_resample(Train_X, Train_Y)
    elif over_sampling == 'RANDOM':
        oversample = RandomOverSampler() 
        Train_X, Train_Y = oversample.fit_resample(Train_X, Train_Y)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.

    # 需要做一个数据转换，转换成np格式.
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
    # path = r'data\0HP Binary Classification'
    path = r'data\0HP'
    generated = 'generated_data/ratio_10_RS'
    # generated = None
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                gan_data=generated,
                                                                length=896,
                                                                number=1000,
                                                                normalization='minmax',
                                                                rate=[0.5, 0.25, 0.25],
                                                                sampling = 'order', 
                                                                over_sampling = 'RANDOM',
                                                                imbalance_ratio = 50,
                                                                )