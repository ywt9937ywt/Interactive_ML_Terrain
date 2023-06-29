import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import cv2
import glob, random
from os.path import exists, join, isdir
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    # return cv2.imwrite(image_path, images)
    return imsave(images, size, image_path)

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size))
    image = np.squeeze(images) * 255 
    # if image.mean() <= 1:
    #     image = np.clip(image, 0, 1) * 255
    # if image.mean() > 1:
    #     image = np.clip(image, 0, 255)
    image = image.astype(int)
    return cv2.imwrite(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, duration=200)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
dim = 128
water_map = np.zeros((dim+2, dim+2))
water_map.fill(-1)
dir_dict = np.zeros((dim+2, dim+2))
is_outlet = np.zeros((dim+2, dim+2))
def river_network():
    global water_map
    global dir_dict
    global is_outlet
    global dim
    # imgpath_list = glob.glob(join("C:\\Users\\10178\\Desktop\\testm", "*_h.png"))
    imgpath_list = [join("C:\\Users\\10178\\Desktop\\testm", "1809_h.png")]
    img_list = [cv2.imread(i, 0) for i in imgpath_list]
    downsample_image = [cv2.resize(i, (dim,dim), interpolation=cv2.INTER_LINEAR) for i in img_list]
    padding_images = [cv2.copyMakeBorder(i, 1,1,1,1, borderType=cv2.BORDER_CONSTANT, value=1000) for i in downsample_image]

    for img in padding_images:
        dir_dict = np.zeros((dim+2, dim+2))
        is_outlet = np.zeros((dim+2, dim+2))
        water_map = np.zeros((dim+2, dim+2))
        water_map.fill(-1)
        for i in range(1, dim+1):
            for j in range(1, dim+1):
                patch1 = np.array(img[i-1: i+2, j-1:j+2])
                patch2 = np.empty((3,3))
                patch2.fill(img[i][j])
                height_inv = patch2 - patch1
                dir = np.argmax(height_inv)
                dir_dict[i][j] = dir+1
                if(np.max(height_inv)== 0):
                    dir_dict[i][j] = 5
        
        for i in range(1, dim+1):
            for j in range(1, dim+1):
                flag = 1
                if(dir_dict[i-1][j-1] == 9 or dir_dict[i-1][j] == 8 or dir_dict[i-1][j+1] == 7 or dir_dict[i][j-1] == 6 or dir_dict[i][j+1] == 4
                   or dir_dict[i+1][j-1] == 3 or dir_dict[i+1][j] == 2 or dir_dict[i+1][j+1] == 1):
                    flag = 0
                else:
                    water_map[i][j] = 1
                is_outlet[i][j] = flag
        # print(water_map)
        for i in range(1, dim+1):
            for j in range(1, dim+1):
                if(water_map[i][j] == -1):
                    calculate_network(i, j)

        # print(water_map[20:25, 20:25])
        # x, y = (water_map > 10).nonzero()
        sketch_sim = water_map
        sketch_sim = np.where(sketch_sim<15, 0, 255)
        # sketch_sim[x, y] = 255
        sketch_sim = sketch_sim[1:-1, 1:-1]
        imsave(sketch_sim, dim, join("C:\\Users\\10178\\Desktop\\testm", "1809_s.png"))
                
def calculate_network(i, j):
    global water_map
    global dir_dict
    global is_outlet
    if(water_map[i][j] != -1):
        return water_map[i][j]
    count = 0
    if(dir_dict[i-1][j-1] == 9):
        count = count + calculate_network(i-1, j-1)
    if(dir_dict[i-1][j] == 8):
        count = count + calculate_network(i-1, j)  
    if(dir_dict[i-1][j+1] == 7):
        count = count + calculate_network(i-1, j+1)  
    if(dir_dict[i][j-1] == 6):
        count = count + calculate_network(i, j-1)  
    if(dir_dict[i][j+1] == 4):
        count = count + calculate_network(i, j+1)  
    if(dir_dict[i+1][j-1] == 3):
        count = count + calculate_network(i+1, j-1)  
    if(dir_dict[i+1][j] == 2):
        count = count + calculate_network(i+1, j)
    if(dir_dict[i+1][j+1] == 1):
        count = count + calculate_network(i+1, j+1)
    water_map[i][j] = count + 1
    return count + 1

