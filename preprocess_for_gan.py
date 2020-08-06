from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


def prepro(d_path, target='B007',length=896, number=1000, normalization='minmax', rate=[0.5, 0.25, 0.25], sampling = 'order', imbalance_ratio = 10):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param target: 需要gan来扩增的数据-关键字
    :param length: 信号长度，默认约2个信号周期，896
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normalization: 数据变换方式: 最大最小值归一化'minmax', 均值归一化'mean'
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param sampling: 训练数据的采样方式. 顺序采样'order', 随机采样'random', 重叠采样'enc'
    :param imbalance_ratio: 某一种大类数据与小类数据的比例,默认为10:1
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess_for_gan.prepro(d_path=path,
                                                                                    target='B007',
                                                                                    length=896,
                                                                                    number=1000,
                                                                                    normalization='minmax',
                                                                                    rate=[0.5, 0.25, 0.25],
                                                                                    sampling='order',
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

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单条数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        ratio = imbalance_ratio # 大类数据和小类数据的不平衡比
        for key in keys:
            slice_data = data[key]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = number  # 样本数
            end_index = end_index // ratio + 2 * length
            Train_sample = []
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
            
            if normalization == 'minmax':
                for i in range(np.size(Train_sample,0)):
                    Train_sample[i] = 2*(Train_sample[i]- min(Train_sample[i]))/(max(Train_sample[i])-min(Train_sample[i]))-1
            elif normalization == 'mean':
                for i in range(np.size(Train_sample,0)):
                    Train_sample[i] = (Train_sample[i]- np.mean(Train_sample[i]))/(max(Train_sample[i])-min(Train_sample[i]))
            elif normalization == '0-1':
                for i in range(np.size(Train_sample,0)):
                    Train_sample[i] = (Train_sample[i]- min(Train_sample[i]))/(max(Train_sample[i])-min(Train_sample[i]))
              
                       
            Train_Samples[key] = Train_sample
        return Train_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        # label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [i] * lenx
            # label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        return Train_Y

    enc_step = 28
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    # gan模式只从小类(故障)文件中读取数据
    filenames = [i for i in filenames if (target in i) and ('.mat' in i)]
    # 从.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 从小类数据集中获得训练数据
    train= slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 需要做一个数据转换，转换成np格式.
    Train_X = np.asarray(Train_X)
    return Train_X, Train_Y


if __name__ == "__main__":
    path = r'data\0HP'
    # minority_sample_filename = '12k_Drive_End_B007_0_118.mat'
    train_X, train_Y = prepro(d_path=path,
                            target='IR021',
                            length=896,
                            number=5,
                            normalization='minmax',
                            rate=[0.5, 0.25, 0.25],
                            sampling='order',
                            imbalance_ratio = 100
                            )