import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
path = r'samples\WGAN-GP\ORDER\ratio-50\orderOR021-05-17-20_23\w-ratio50-order-minmax-OR021_98331.out'
f_path = path.split('.')[0]
file = open(path)
list_arr = file.readlines()
l = len(list_arr)
for i in range(l): 
    list_arr[i] = list_arr[i].split(',')
    for str in range(len(list_arr[i])):
        list_arr[i][str] = list_arr[i][str].split( )
file.close()
a = np.array(list_arr)
epoch = np.shape(a)[0]
num = 1
d_loss = a[num:epoch,1,1]
d_loss = d_loss.astype(float)
g_loss = a[num:epoch,2,1]
g_loss = g_loss.astype(float)

fig = plt.figure()
# gs = gridspec.GridSpec(2,2)
    # gs.update(wspace = 0.05, hspace = 0.05)
    # samples = np.reshape(samples, [num,length])
x = np.arange(num, epoch)
x *= 100
    # ax = plt.subplot(gs[i%2,i//2])
    # y = samples[i]
y1 = plt.plot(x, d_loss,'g', label = 'd_loss')
y2 = plt.plot(x, g_loss,'r', label = 'g_loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
name = f_path.split('\\')
name = name[-1]
plt.title(name)
plt.show()
fig.savefig('{}.png'.format(f_path))
plt.close()
# epoch = np.shape(s)[0]


# print(s)