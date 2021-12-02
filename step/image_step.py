import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch import nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image
from transform.basic_transform import ScaleResize
from torchvision import transforms
import config.train_config
import dataset.image_dataset
import utils.util_data
from itertools import cycle
import argparse
import time
import warnings
import gc
from transform.image_transform import ImageTestTransformOneRaw

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

cfg = config.train_config.TrainConfig()
warnings.filterwarnings("ignore")
cudnn.deterministic = False
cudnn.benchmark = True
# cudnn.benchmark=False # if benchmark=True, deterministic will be False
# cudnn.deterministic = True
torch.backends.cudnn.enabled = True


# torch.backends.cudnn.deterministic = True

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ImageStep():
    def __init__(self, args):
        super().__init__()
        self.path = {}
        self.path['root'] = args.root
        self.path['result_path'] = os.path.join(self.path['root'], args.result_dir)
        if not os.path.exists(self.path['result_path']):
            os.makedirs(self.path['result_path'])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_class = args.num_class
        self.model_name = args.model_name
        self.opt = args.opt
        self.lr = args.lr
        self.start_epoch = args.start_epoch
        self.epoch_num = args.epoch_num
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.image_size = args.image_size
        self.image_channels = args.image_channels
        self.batch_size = args.batch_size
        self.list = [[], [], [], [], []]
        self.train_test = args.train_test
        self.device_ids = [0, 1, 2]
        if self.train_test == 'train':
            self.save_hparam(args)

    def model_zoo(self):
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(self.image_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                               padding=(3, 3), bias=False)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(self.image_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                               padding=(3, 3), bias=False)
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(self.image_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                               padding=(3, 3), bias=False)
        elif self.model_name == 'densenet121':
            self.model = models.densenet121(pretrained=False, num_classes=self.num_class)
            self.model.features.conv0 = torch.nn.Conv2d(self.image_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                                        padding=(3, 3), bias=False)

    def load_init_model(self):
        self.model_zoo()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.to(self.device)

        self.loss_func = nn.CrossEntropyLoss()
        if self.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print(self.model)

    def load_latest_epoch(self):
        self.model_zoo()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.to(self.device)

        if self.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        load_path = self.path['result_path'] + self.model_name + '_model_G' + str(self.image_size) + '_' + str(
            self.start_epoch - 1) + '.pth'
        list_path = self.path['result_path'] + self.model_name + '_list_G' + str(self.image_size) + '_' + str(
            self.start_epoch - 1) + '.bin'
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_func = checkpoint['loss']
        self.list = torch.load(list_path)

    def load_trained_model(self):
        # original saved file with DataParallel
        self.model_zoo()
        load_path = self.path['result_path'] + self.model_name + '_model_G' + str(self.image_size) + '_' + str(
            self.epoch_num - 1) + '.pth'
        loaded_dict = torch.load(load_path)

        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.to(self.device)
        # print(loaded_dict['model_state_dict'])
        self.model.load_state_dict(loaded_dict['model_state_dict'])
        self.model.eval()

    def model_one_GPU(self):
        self.model_zoo()
        load_path = self.path['result_path'] + self.model_name + '_model_G' + str(self.image_size) + '_' + str(
            self.epoch_num - 1) + '.pth'
        loaded_dict = torch.load(load_path)

        # if not the DataParallel model
        new_state_dict = OrderedDict()
        for k, v in loaded_dict['model_state_dict'].items():
            namekey = k[7:] if k.startswith('module.') else k
            new_state_dict[namekey] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)

    def load_data(self, **kwargs):
        bank = {}
        doc = {}
        bank['train'] = dataset.image_dataset.Bank_Train(cfg)
        bank['val'] = dataset.image_dataset.Bank_Val(cfg)
        bank['test'] = dataset.image_dataset.Bank_Test(cfg)
        doc['train'] = dataset.image_dataset.Doc_Train(cfg)
        doc['val'] = dataset.image_dataset.Doc_Val(cfg)
        doc['test'] = dataset.image_dataset.Doc_Test(cfg)

        bank_loader = utils.generate_data_loaders(bank)
        doc_loader = utils.generate_data_loaders(doc)

        if kwargs['dataset'] == 'bank_doc_train':
            self.train_loader_bank = bank_loader['train']
            self.test_loader_bank = bank_loader['test']
            self.train_loader_doc = doc_loader['train']
            self.test_loader_doc = doc_loader['test']
            self.val_loader_bank = bank_loader['val']
            self.val_loader_doc = doc_loader['val']

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_DNN(self):
        # torch.cuda.empty_cache()
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        Loss, Error = 0, 0

        for i, (bank_data, doc_data) in enumerate(zip(cycle(self.train_loader_bank), self.train_loader_doc)):
            # 4*4*2
            bank_images = bank_data['image']
            bank_images = bank_images.view([-1, self.image_channels, self.image_size, self.image_size])
            doc_images = doc_data['image']
            doc_images = doc_images.view([-1, self.image_channels, self.image_size, self.image_size])
            images = torch.concat((bank_images, doc_images), 0)
            # print(images.shape)
            # If batch_size == 4, the shape of input data is: [32,1,1300,1300],
            # where the  32 means 4 batch_size with 4 rotation angles --> 16
            # bank and doc data --> 16 * 2 == 32
            bank_flags = bank_data['rotate_flag']
            bank_flags = bank_flags.view([-1, 1])
            bank_flags = bank_flags.squeeze()
            doc_flags = doc_data['rotate_flag']
            doc_flags = doc_flags.view([-1, 1])
            doc_flags = doc_flags.squeeze()
            rotated_flags = torch.concat((bank_flags, doc_flags), 0)

            images = images.float()
            images = images.to(self.device)
            rotated_flags = rotated_flags.to(self.device)
            # print('images shape: {}'.format(images.shape))
            # [4,3,1300,1300]
            output = self.model(images)
            loss = self.loss_func(output, rotated_flags)
            # print(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pre = output.detach().max(1)[1]
            error = self.get_error(pre, rotated_flags)
            Error += error
            Loss += loss.cpu().item()

            if i % 10 == 0:
                print("[train] batch: %d, loss: %.3f, error: %.3f" % (i + 1, Loss / (i + 1), Error / (i + 1)))

            # break
        self.list[0].append(Loss / (i + 1))
        self.list[1].append(Error / (i + 1))

    def val_DNN(self):
        self.model.eval()
        Error = 0
        with torch.no_grad():
            for i, (bank_data, doc_data) in enumerate(zip(cycle(self.val_loader_bank), self.val_loader_doc)):
                bank_images = bank_data['image']
                bank_images = bank_images.view([-1, self.image_channels, self.image_size, self.image_size])
                doc_images = doc_data['image']
                doc_images = doc_images.view([-1, self.image_channels, self.image_size, self.image_size])
                images = torch.concat((bank_images, doc_images), 0)

                bank_flags = bank_data['rotate_flag']
                bank_flags = bank_flags.view([-1, 1])
                bank_flags = bank_flags.squeeze()
                doc_flags = doc_data['rotate_flag']
                doc_flags = doc_flags.view([-1, 1])
                doc_flags = doc_flags.squeeze()
                rotated_flags = torch.concat((bank_flags, doc_flags), 0)

                images = images.float()
                images = images.to(self.device)
                rotated_flags = rotated_flags.to(self.device)
                output = self.model(images)
                # loss = self.loss_func(output, rotated_flags)
                pre = output.detach().max(1)[1]
                error = self.get_error(pre, rotated_flags)

                Error += error
                if i % 10 == 0:
                    print("[val] batch: %d, error: %.3f" % (i + 1, Error / (i + 1)))

                # break
            self.list[2].append(Error / (i + 1))

    def work_train(self, args):
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epoch_num)
        for epoch in range(self.start_epoch, self.epoch_num):
            print("epoch %d:" % (epoch + 1))
            tis = time.time()
            # adjust learning rate
            if args.logspace != 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]

            self.train_DNN()
            self.val_DNN()

            # 0 : epoch_num-1
            # if epoch != 0 and epoch % 2 == 0 or epoch == self.epoch_num -1:
            self.save_latest_epoch(epoch)

            time_expand = time.time() - tis
            print("TIME CMD: [{}] s".format(time_expand))
            time_remain = time_expand * (self.epoch_num - epoch - 1)
            time_scale = 's'
            if time_remain > 60:
                time_scale = 'min'
                time_remain = time_remain / 60
            print("Approximate reamaining time: [{}] {}\n".format(time_remain, time_scale))

        self.draw_figures()
        self.save_list_val()

    # test DNN on all test dataset
    def work_test_all(self):
        # self.model.eval()
        Error = 0
        with torch.no_grad():
            for i, (bank_data, doc_data) in enumerate(zip(cycle(self.test_loader_bank), self.test_loader_doc)):
                # because test_doc is longer than test_bank
                bank_images = bank_data['image']
                bank_images = bank_images.view([-1, self.image_channels, self.image_size, self.image_size])
                doc_images = doc_data['image']
                doc_images = doc_images.view([-1, self.image_channels, self.image_size, self.image_size])
                images = torch.concat((bank_images, doc_images), 0)

                bank_flags = bank_data['rotate_flag']
                bank_flags = bank_flags.view([-1, 1])
                bank_flags = bank_flags.squeeze()
                doc_flags = doc_data['rotate_flag']
                doc_flags = doc_flags.view([-1, 1])
                doc_flags = doc_flags.squeeze()
                if bank_flags.ndim == 0:
                    bank_flags = bank_flags.unsqueeze(0)
                if doc_flags.ndim == 0:
                    doc_flags = doc_flags.unsqueeze(0)
                rotated_flags = torch.concat((bank_flags, doc_flags), 0)

                images = images.float()
                images = images.to(self.device)
                rotated_flags = rotated_flags.to(self.device)
                tis = time.time()
                output = self.model(images)
                time_expand = time.time() - tis

                pre = output.detach().max(1)[1]
                error = self.get_error(pre, rotated_flags)

                Error += error
                if i % 10 == 0:
                    print("[test all] batch: %d, error: %.3f" % (i + 1, Error / (i + 1)))
                    self.list[3].append(Error / (i + 1))

            print("[TIME] batch size: %d, time: %.5fs" % (self.batch_size * 8, time_expand))

            self.list[3].append(Error / (i + 1))
            test_error = self.list[3]
            print('Error on test_all dataset: {}'.format(test_error[-1]))
            accuracy = (1 - test_error[-1]) * 100
            print('Accuracy on test_all dataset: {}%'.format(accuracy))
            self.save_test_all()
    # [TIME] batch size: 32, time: 0.12558s (resnet18)

    # test DNN on bank test dataset only
    def work_test_bank(self):
        Error = 0
        i = 0
        time_expand = 0
        with torch.no_grad():
            for bank_data in self.test_loader_bank:
                bank_images = bank_data['image']
                bank_images = bank_images.view([-1, self.image_channels, self.image_size,self.image_size])

                bank_flags = bank_data['rotate_flag']
                bank_flags = bank_flags.view([-1, 1])
                rotated_flags = bank_flags.squeeze()

                images = bank_images.float()
                images = images.to(self.device)
                rotated_flags = rotated_flags.to(self.device)
                tis = time.time()
                output = self.model(images)
                time_expand = time.time() - tis
                pre = output.detach().max(1)[1]
                error = self.get_error(pre, rotated_flags)

                Error += error
                if i % 10 == 0:
                    print("[test bank] batch: %d, error: %.3f" % (i + 1, Error / (i + 1)))
                    # print("[test bank] batch size: %d, time: %.5f" % (self.batch_size*4, time_expand))
                    self.list[4].append(Error / (i + 1))
                i += 1

            print("[TIME] batch size: %d, time: %.5fs" % (self.batch_size * 8, time_expand))

            self.list[4].append(Error / i)
            test_error = self.list[4]
            print('Error on test_bank dataset: {}'.format(test_error[-1]))
            accuracy = (1 - test_error[-1]) * 100
            print('Accuracy on test_bank dataset: {}%'.format(accuracy))
            self.save_test_bank()
    # [TIME] batch size: 32, time: 0.13804s (result7/resnet18 256*256)
    # [TIME] batch size: 32, time: 0.77109s (result3/resnet34 1000*1000)

    # test the infer time on one image
    def test_one_image(self):
        # self.model().eval()

        image = Image.open('/home/std2022/zhaoxu/tmp.jpg').convert('RGB')
                           # 'disk/Bank/Images/0047.jpg').convert('RGB')
        transform = transforms.Compose([
            # ScaleResize(fixed_size=(1000,1000), fill_value=(255, 255, 255)),
            # Change image channels from 3 to 1
            transforms.Grayscale(1),

            transforms.ToTensor()
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        print('tmp Image Shape:{}'.format(image.shape))

        image = image.to(self.device)
        # print(image.shape)

        # tis = time.time()
        output = self.model(image)
        pre = output.detach().max(1)[1]
        # time_expand = time.time() - tis

        print("predict: {}".format(pre))
        # print("time used on one image: {}s".format(time_expand))

        # +++++ Resize to 1000*1000 +++++
        # predict: tensor([0], device='cuda:0')
        # time used on one image: 0.6052532196044922

        # model path: result5/resnet34_..._E4
        # predict: tensor([0], device='cuda:0')
        # time used on one image: 0.37607359886169434
        # just time used on predict
        # ------------------------
        # predict: tensor([0], device='cuda:0')
        # time used on one image: 0.4475886821746826
        # timed used on the whole process

        # model path: result7/resnet18...E4
        # predict: tensor([0], device='cuda:0')
        # time used on one image: 0.3287224769592285s

        # model path: result3/resnet34_..._E7
        # time used on one image: 0.4780392646789551s
        # just time used on predict

        # +++++ Use the original size +++++
        # predict: tensor([2], device='cuda:0')
        # Time used on one image completely: 0.64734 s

    # def get_error(self, pre, lbl):
    #     acc = torch.sum(pre == lbl).item() / len(pre)
    #     return 1 - acc

    def test_original_image(self):
        test_transform = ImageTestTransformOneRaw()
        correct_num = 0
        total = 0
        for filename in os.listdir('/home/std2022/zhaoxu/disk/Bank/Images/'):
            image = Image.open('/home/std2022/zhaoxu/disk/Bank/Images/'+filename).convert('RGB')
            image, flag = test_transform(image)
            # 输入模型
            image = torch.unsqueeze(image,0)
            image = image.to(self.device)
            flag = flag.to(self.device)
            # print(image.shape)
            tis = time.time()
            output = self.model(image)
            pre = output.detach().max(1)[1]
            time_expand = time.time() - tis
            # print(flag)
            # print(pre)
            # 评估准确率
            if pre == flag:
                correct_num += 1
            print("{} time used on the image: {:.4f}s".format(total,time_expand))
            total += 1

            if total == 500:
                break
            # torch.cuda.empty_cache()
            # gc.collect()
            #  CUDA out of memory.

        accuracy = correct_num/total
        accuracy = accuracy * 100
        print("Accuracy on the original image: {:.5f} %".format(accuracy))
        # Accuracy on the original image: 75.20000 %

    def draw_figures(self):
        x = np.arange(0, len(self.list[0]), 1)
        train_l, train_e, val_e = np.array(self.list[0]), np.array(self.list[1]), \
                                  np.array(self.list[2])
        plt.figure()
        ax1 = plt.subplot(211)
        ax1.set_title('train loss')
        plt.plot(x, train_l, color='red', label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        ax2 = plt.subplot(212)
        ax2.set_title('train and val error')
        plt.plot(x, train_e, color='red', label='train error')
        plt.plot(x, val_e, color='blue', label='val error')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend(loc='best')
        plt.subplots_adjust(hspace=0.5)

        save_path = self.path['result_path'] + self.model_name + '_G' + str(self.image_size) + '_E' + str(
            self.epoch_num) + '_curve.png'
        plt.savefig(save_path)
        plt.close()

    def save_latest_epoch(self, epoch):
        save_path = self.path['result_path'] + self.model_name + '_model_G' + str(self.image_size) + '_' + str(
            epoch) + '.pth'
        list_path = self.path['result_path'] + self.model_name + '_list_G' + str(self.image_size) + '_' + str(
            epoch) + '.bin'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_func,
        }, save_path)
        torch.save(self.list, list_path)
        print("Successfully save model {}, epoch {}".format(self.model_name, epoch))

    # save hyper_param
    def save_hparam(self, args):
        save_path = self.path['result_path'] + self.model_name + '_G' + str(self.image_size) + '_E' + str(
            self.epoch_num) + '_hparam.txt'
        with open(save_path, 'w') as fp:
            args_dict = args.__dict__
            for key in args_dict:
                fp.write("{} : {}\n".format(key, args_dict[key]))

    def save_list_val(self):
        save_path = self.path['result_path'] + self.model_name + '_loss_G' + str(self.image_size) + '_Ep' + str(
            self.epoch_num) + '_train_val.txt'
        with open(save_path, 'w') as fp:
            train_loss = self.list[0]
            train_error = self.list[1]
            val_error = self.list[2]

            for i in range(len(train_loss)):
                fp.write('epoch:{}, train_loss:{}, train_error:{}, val_error:{}\n'.format(i + 1,
                                                                                          train_loss[i], train_error[i],
                                                                                          val_error[i]))
            print("Successfully save train val lists!")

    def save_test_all(self):
        save_path = self.path['result_path'] + self.model_name + '_G' + str(1000) + '_E' + str(
            self.epoch_num) + '_test_all.txt'
        save_path = os.path.join(save_path)
        with open(save_path, 'w') as fp:
            test_error = self.list[3]
            title = self.model_name + '_epoch_' + str(self.epoch_num) + '\n'
            fp.write(title)
            fp.write('test error on ALL dataset\n')
            for i in range(len(test_error) - 1):
                accuracy = (1 - test_error[i]) * 100
                fp.write('batch:{}, test_error:{}, test_accuracy:{}%\n'.format(10 * i + 1, test_error[i], accuracy))
            accuracy = (1 - test_error[-1]) * 100
            fp.write('ALL dataset: test_error:{}, test_accuracy:{}%\n'.format(test_error[-1], accuracy))
        print("Successfully save test error and accuracy[ALL]!")

    def save_test_bank(self):
        save_path = self.path['result_path'] + self.model_name + '_G' + str(self.image_size) + '_E' + str(
            self.epoch_num) + '_test_bank.txt'
        save_path = os.path.join(save_path)
        with open(save_path, 'w') as fp:
            test_error = self.list[4]
            title = self.model_name + '_epoch_' + str(self.epoch_num) + '\n'
            fp.write(title)
            fp.write('test error on BANK dataset\n')
            for i in range(len(test_error) - 1):
                accuracy = (1 - test_error[i]) * 100
                fp.write('batch:{}, test_error:{}, test_accuracy:{}%\n'.format(10 * i + 1, test_error[i], accuracy))
            accuracy = (1 - test_error[-1]) * 100
            fp.write('BANK dataset: test_error:{}, test_accuracy:{}%\n'.format(test_error[-1], accuracy))
        print("Successfully save test error and accuracy[BANK]!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/std2022/zhaoxu/')
    parser.add_argument('--result-dir', default='disk/result/result3/')
    parser.add_argument('--train-test', default='test', type=str, help='choose to train or test model')
    parser.add_argument('--model-name', default='resnet34', type=str, help='select the model')
    parser.add_argument('--start-epoch', default=0, type=int, help='whether to train the model from scratch')
    parser.add_argument('--epoch-num', default=7, type=int, help='epoch numbers')  # 3
    parser.add_argument('--opt', default='sgd', type=str, help='select the optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')  # 0.001
    parser.add_argument('--logspace', default=1, type=float, help='adjust learning rate, refer to def work_train()')
    parser.add_argument('--momentum', default=0.9, type=float, help='params of optimizer: momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='params of optimizer: weight decay')
    parser.add_argument('--num-class', default=4, type=int, help='number of classes')
    parser.add_argument('--image-size', default=cfg.IMAGE_SIZE[0], type=int, help='change the image size(square)')
    parser.add_argument('--image-channels', default=1, type=int, help='the channels of image: 1 or 3')
    parser.add_argument('--batch-size', default=cfg.BATCH_SIZE, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu-id', default='0,1,3', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    seed_num = args.epoch_num + args.seed
    seed_torch(seed_num)

    trainer = ImageStep(args)

    # "train" indicates training model, while "test" indicates test_DNN
    # if args.train_test == 'train':
    #     # start_epoch ==0, train the model from scratch
    #     if args.start_epoch == 0:
    #         trainer.load_init_model()
    #         trainer.load_data(dataset='bank_doc_train')
    #         trainer.work_train(args)
    #     else:
    #         trainer.load_latest_epoch()
    #         trainer.load_data(dataset='bank_doc_train')
    #         trainer.work_train(args)
    #
    # elif args.train_test == 'test':
    #     # Test the model on all dataset
    #     trainer.load_trained_model()
    #     trainer.load_data(dataset='bank_doc_train')
    #     # trainer.work_test_all()  # test DNN on all (bank and doc) test dataset
    #     trainer.work_test_bank()  # test DNN on bank test dataset

    # Test the model on one image
    trainer.model_one_GPU()
    trainer.test_original_image()
    # tis = time.time()
    # trainer.test_one_image()
    # time_expand = time.time()-tis
    # print("Time used on one image completely: %.5f s" % (time_expand))

if __name__ == '__main__':
    main()

# module....
# OrderedDict([('conv1.weight', tensor([[[[ 0.1216,  0.0266, -0.1001,  ...,  0.0765, -0.0293, -0.1017],
# Missing key(s) in state_dict: "module.conv1.weight"
# Unexpected key(s) in state_dict: "conv1.weight"
# model.eval()位置

'''
ResNet50
(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
(fc): Linear(in_features=2048, out_features=4, bias=True)

ResNet34
(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 (fc): Linear(in_features=512, out_features=4, bias=True)
 
batch_size: 4 
image_size: 1400*1400
loss: 1.281 error: 0.575

[test] batch: 141, loss: 1.336, error: 0.615 (after training)
[test] batch: 141, loss: 1.514, error: 0.734 (no train)

batch_size: 4
image_size: 1600*1600
loss: 1.353 error:0.541

image_size: 1300*1300
[train] batch: 1591, loss: 0.942, error: 0.515

batch_size: 8
image_size: 1000*1000
loss: 1.381 error: 0.706

ResNet18
batch_size: 4
image_size: 1600*1600
loss: 1.366 error: 0.587

batch_size: 8
image_size: 1000*1000
loss: 1.385 error 0.720
'''

'''
ResNet50 --> worse than ResNet34
DenseNet --> worse than Resnet
--------------------------------
Conclusion: simpler models can achieve better performance on this task.

resize to 256*256 --> decrease the accuracy

change to 3 channels --> little influence
'''

# Train with image size 1000*1000, use the 1400*1400 image to test  --> 效果不如用1000*1000 的test
# 逐个测试图片