from typing import Iterator, Optional, Sized
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import glob
import h5py
from matplotlib.animation import FuncAnimation
from torch.autograd import grad
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from os.path import exists, join, isdir
from os import makedirs

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 8) * (self.input_size // 8)),
            nn.BatchNorm1d(128 * (self.input_size // 8) * (self.input_size // 8)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 8), (self.input_size // 8))
        x = self.deconv(x)

        return (x + 1)/2

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            # nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class WGAN(object):
    def __init__(self, config):
        # parameters
        self.config = config
        self.epoch = config.epoch
        self.sample_num = 1000
        self.batch_size = config.nimg_batch
        self.save_dir = join(config.dataset_root_path, config.save_dir)
        self.result_dir = join(config.dataset_root_path, config.result_dir)
        # self.log_dir = args.log_dir
        self.gpu_mode = config.gpu_mode
        self.model_name = config.gan_type
        self.input_size = config.input_size
        self.z_dim = 62
        self.c = 0.01                   # clipping value
        self.lambda_ = 10
        self.n_critic = 5               # the number of iterations of the critic per generator iteration

        self.dataset = WGAN_dataset(self.config)
        training_sampler = WGAN_Sampler(self.dataset, self.config)
        training_batch_sampler = WGANBatchSampler(training_sampler, self.config)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size= 1,
                                      collate_fn=WGAN_Collate, 
                                    #   sampler = training_sampler,
                                      batch_sampler=training_batch_sampler)
        data = self.dataset[0]
        # print(np.array(data).shape)

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size)
        self.D = discriminator(input_dim=1, output_dim=1, input_size=self.input_size)
        # self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        # self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config.lrG, betas=(config.beta1, config.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=config.lrD, betas=(config.beta1, config.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

        # g = self.G(self.sample_z_.float().cuda())
        # p = 'E:\\terrain_project\\train_results\\WGAN_epoch001.png'
        # g = np.squeeze(g.cpu().detach().numpy()[0])
        # print(g.shape)
        # utils.save_images(g, [1,1], p)
        # plt.imshow(g, cmap='gray')
        # plt.show()
        # g = (np.clip(g, 0, 1) * 255).astype(int)
        # print(g)
        # cv2.imwrite(p, g)
        # d = self.D(g)
        # print(d.cpu().detach().shape)

    def train(self):

        self.train_hist = {}
        self.meanval = []
        self.minval = []
        self.maxval = []
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            count = 0
            for iter, x_ in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_.float())
                D_real_loss = torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                alpha = torch.rand((self.batch_size, 1, 1, 1))
                if self.gpu_mode:
                    alpha = alpha.cuda()

                x_hat = alpha * x_.data + (1 - alpha) * G_.data
                x_hat.requires_grad = True

                pred_hat = self.D(x_hat.float())
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = -D_real_loss + D_fake_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

                # clipping D
                # for p in self.D.parameters():
                #     p.data.clamp_(-self.c, self.c)

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.item())

                count = count + 1
                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            # print("count {0}".format(count))
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if epoch % 3 == 0:
                with torch.no_grad():
                    self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.model_name,
                                 self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)

        r = range(self.epoch)
        plt.title('val')
        plt.plot(r, self.meanval, 'r-')
        plt.plot(r, self.maxval, 'b-')
        plt.plot(r, self.minval, 'g-')
        plt.show()

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        # samples = (samples + 1) / 2
        self.firsttime = True
        img = np.squeeze(samples[0])
        # self.plt_image = img
        self.meanval.append(np.mean(img))
        self.minval.append(np.min(img))
        self.maxval.append(np.max(img))

        #real time plot image
        # ax1 = plt.subplot(1,2,1)
        # self.real_time_plt1 = ax1.imshow(img)
        # ani = FuncAnimation(plt.gcf(), self.update, interval=200, cache_frame_data=False)
        # plt.ion()
        # plt.cla()
        # plt.figure()
        # plt.imshow(img, cmap="gray")
        # if self.firsttime:
        #     # plt.show()
        #     plt.show(block = False)
        #     self.firsttime = False
        # else:
        #     plt.draw()
        #     plt.pause(0.1)
        # plt.show()
        # print("after plot")
        # utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :][0], [image_frame_dim, image_frame_dim],
        #                   self.result_dir + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
        utils.save_images(samples[0], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def update(self, frame):
        plt.cla()
        plt.plot(self.plt_image)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

class WGAN_dataset(Dataset):
    def __init__(self, config, mode='train'):
        print("--------Initialize WGAN dataset --------")
        self.config = config
        self.filename_list = ['pre_128.npy']
        self.convert(self.filename_list[0])
        self.dataset_seg = [0.8, 0.1, 0.1]
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])
        # load converted data
        f = h5py.File(self.my_hdfs)
        self.allimg = np.array(f[self.filename_list[0]])
        # print(np.array(self.allimg).shape)
        shuffled_idx = np.random.permutation(len(self.allimg))

        #train, validate. test segmentation
        index = int(len(self.allimg) * self.dataset_seg[0])
        self.trainset = self.allimg[:index]
        index2 = int(len(self.allimg) * self.dataset_seg[2])
        self.valset = self.allimg[index:-index2]
        self.testset = self.allimg[-index2:]
        print("--------Initializing WGAN dataset finished--------")

    def __len__(self):
        if self.mode == 'train':
            return len(self.trainset)
        if self.mode == 'validate':
            return len(self.valset)
        if self.mode == 'test':
            return len(self.testset)
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            # return np.transpose(self.trainset[idx], axes=[1,0,2,3])
            # print(np.array(self.trainset[idx]).shape)
            return self.trainset[idx]
        if self.mode == 'validate':
            # return np.transpose(self.valset[idx], axes=[1,0,2,3])
            return self.valset[idx]
        if self.mode == 'test':
            # return np.transpose(self.testset[idx], axes=[1,0,2,3])
            return self.testset[idx]

    def convert(self, filename, rewrite = True):
        imagepath = join(self.config.dataset_root_path, 'terrain_data2128')
        hdfs_container = join(self.config.dataset_root_path, self.config.hdf_path)
        self.my_hdfs = join(hdfs_container, 'terrain2128_hdf.h5')
        if not exists(imagepath):
            print("data files not exist")
            return
        
        if exists(self.my_hdfs) and not rewrite:
            return
        img_list = glob.glob(join(imagepath, "*_h.png"))
        shape = np.array(cv2.imread(img_list[0], 0)).shape
        num_img = len(img_list)
        mydataset = np.zeros((num_img, shape[0], shape[1]))
        
        for idx, img_path in enumerate(img_list):
            height_map = cv2.imread(join(imagepath, str(idx+1).zfill(4) + "_h.png"), 0)
            mydataset[idx, :, :] = height_map.astype(float) / 255
        
        np.save(join(self.config.dataset_root_path, filename), mydataset)
        print("----------{0} generated ---------".format(filename))

        
        with h5py.File(self.my_hdfs, 'w') as f:
            f[filename] = mydataset

    
class WGAN_Sampler(Sampler):
    def __init__(self, dataset, config, mode='train'):
        self.config = config
        self.mode = mode
        self.dataset = dataset
        self.datalength = self.dataset.__len__()

    def __iter__(self):
        indices = np.random.choice(self.datalength, self.config.steps_epoch * self.config.nimg_batch)
        return iter(indices)
    
    def __len__(self):
        return self.config.steps_epoch * self.config.nimg_batch
    
class WGANBatchSampler:
    def __init__(self, sampler, config):
        self.sampler = sampler
        self.config = config

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.config.nimg_batch:
                yield batch
                batch = []
            
        if len(batch) > 0 and not self.config.drop_last:
            yield batch

    def __len__(self):
        if self.config.drop_last:
            return len(self.sampler) // self.config.nimg_batch
        else:
            return (len(self.sampler) + self.config.nimg_batch - 1) // self.config.nimg_batch
        
def WGAN_Collate(batch_data):
    # print(np.expand_dims(batch_data, axis=1).shape)
    return torch.from_numpy(np.expand_dims(batch_data, axis=1))
