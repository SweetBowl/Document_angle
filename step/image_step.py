import os
import random
import numpy as np
import torch
import torchvision.models as models
from torch import nn
import torch.backends.cudnn as cudnn
import config.train_config
import dataset.image_dataset
import utils.util_data
from itertools import cycle
import argparse

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

# from step.abstract_step import *

cfg = config.train_config.TrainConfig()

cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ImageStep():
    def __init__(self, args):
        super().__init__()
        self.path = {}
        self.path['root'] = '/home/std2022/zhaoxu/'
        self.path['result_path'] = os.path.join(self.path['root'],args.result_path)
        if not os.path.exists(self.path['result_path']):
            os.makedirs(self.path['result_path'])

        # self.config=args.config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_class = args.num_class
        self.model_name = args.model_name
        self.opt = args.opt
        self.lr = args.lr
        self.start_epoch = args.start_epoch
        self.epoch_num = args.epoch_num
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.save_hparam(args)

    def load_models(self):
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        if self.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

        # print("hello")

        # demo
        if kwargs['dataset'] == 'train':
            self.train_loader_bank = bank_loader['train']
            self.test_loader_bank = bank_loader['test']
            self.train_loader_doc = doc_loader['train']
            self.test_loader_doc = doc_loader['test']
        # elif kwargs['dataset'] == 'bank_train':
        #     self.train_loader = bank_loader['train']

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_DNN(self):

        torch.cuda.empty_cache()

        self.model.train()
        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3])

        torch.autograd.set_detect_anomaly(True)

        Loss, Error = 0, 0
        img_num = 0

        i = 0

        for _,(bank_data,doc_data) in enumerate(zip(cycle(self.train_loader_bank),self.train_loader_doc)):
            #
            bank_images = bank_data['image']
            bank_images = bank_images.view([-1, 1, 1300, 1300])
            doc_images = doc_data['image']
            doc_images = doc_images.view([-1,1,1300,1300])
            images = torch.concat((bank_images,doc_images),0)
            # print(images.shape)
            # break
            bank_flags = bank_data['rotate_flag']
            bank_flags = bank_flags.view([-1, 1])
            bank_flags = bank_flags.squeeze()
            doc_flags = doc_data['rotate_flag']
            doc_flags = doc_flags.view([-1,1])
            doc_flags = doc_flags.squeeze()
            rotated_flags = torch.concat((bank_flags,doc_flags),0)

            images = images.float()
            images = images.to(self.device)
            rotated_flags = rotated_flags.to(self.device)
            # print('images shape: {}'.format(images.shape))
            # [4,3,1600,1600]
            output = self.model(images)
            # print('output shape: {}'.format(output.shape))
            # [4,4]
            # print(i)
            # print(output)
            # i += 1
            # print('rotated_flags shape:{}'.format(rotated_flags.shape))
            # print(rotated_flags)
            img_num = img_num + len(images)
            loss = self.loss_func(output, rotated_flags)
            # print(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(output)
            pre = output.detach().max(1)[1]
            # print('predict: {}'.format(pre))
            error = self.get_error(pre, rotated_flags)
            Error += error
            Loss += loss.item()

            if i % 10 == 0:
                print("[train] batch: %d, loss: %.3f, error: %.3f" % (i + 1, Loss / (i + 1), Error / (i + 1)))

            i += 1

        # test DNN

    def testDNN(self):
        torch.cuda.empty_cache()

        self.model.eval()
        self.model.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3])

        Loss, Error = 0, 0
        i = 0
        for data in self.test_loader_bank:
            images = data['image']
            images = images.view([-1, 1, 1300, 1300])
            rotated_flags = data['rotate_flag']
            rotated_flags = rotated_flags.view([-1, 1])
            rotated_flags = rotated_flags.squeeze()

            images = images.float()
            images = images.to(self.device)
            rotated_flags = rotated_flags.to(self.device)
            output = self.model(images)
            loss = self.loss_func(output, rotated_flags)
            pre = output.detach().max(1)[1]
            error = self.get_error(pre, rotated_flags)

            Error += error
            Loss += loss.item()
            if i % 10 == 0:
                print("[test] batch: %d, loss: %.3f, error: %.3f" % (i + 1, Loss / (i + 1), Error / (i + 1)))
            i += 1

    def work(self, args):
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epoch_num)
        for epoch in range(self.start_epoch, self.epoch_num):
            # adjust learning rate
            if args.logspace != 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]
            self.train_DNN()
            self.testDNN()
            print("epoch: %d\n" % epoch)
        # save model
        self.save_model()

    def save_model(self):
        torch.save(self.path['result_path'],'resnet34.pkl')

        # save param
    def save_hparam(self,args):
        save_path = os.path.join(self.path['result_path'],'hparam.txt')
        with open(save_path,'w') as fp:
            args_dict = args.__dict__
            for key in args_dict:
                fp.write("{} : {}\n".format(key,args_dict[key]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('---model_name', default='resnet34', type=str)
    parser.add_argument('--result_path',default='Document_angle/result/models/')
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--logspace', default=1, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--device',default='0,1,2',type=str)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    trainer = ImageStep(args)
    trainer.load_models()

    trainer.load_data(dataset='train')
    # trainer.train_DNN()
    # trainer.testDNN()
    trainer.work(args)
    trainer.save_hparam(args)

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.current_device())


if __name__ == '__main__':
    main()

# save model
# use doc train

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
