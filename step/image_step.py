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
import argparse

# from step.abstract_step import *

cfg = config.train_config.TrainConfig()
'''
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

print("hello")

index = 0
i = 0

for data in bank_loader['test']:

    images = data['image']
    # (64,4,3,1600,1600)
    images = images.view([-1,3,1600,1600])
    # (256,3,1600,1600)

    rotate_flags = data['rotate_flag']
    # (64,4): [[0,1,2,3],[0,1,2,3],...]
    rotate_flags = rotate_flags.view([-1,1])

    for image, rotate_flag in zip(images, rotate_flags):
        pil_image = F.to_pil_image(image)
        pil_image.save('/home/std2022/zhaoxu/Document_angle/test1/' + str(index) +
                       '_{}'.format(rotate_flag.item()) + '.jpg')

        # print('index_{}_flag{}'.format(index,rotate_flag.item()))
        i += 1
        if i % 4 == 0:
            index += 1
'''

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
        # self.config=args.config
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.num_class = args.num_class
        self.model_name = args.model_name
        self.opt = args.opt
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay

    def load_models(self):
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False, num_classes=self.num_class)

        # print('model: \n')
        # print(self.model)

        self.loss_func = nn.CrossEntropyLoss()
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
        if kwargs['dataset'] == 'bank_test':
            self.test_loader = bank_loader['test']
            # index = 0
            # i = 0
            #
            # for data in bank_loader['test']:
            #     images = data['image']
            #     # (64,4,3,1600,1600)
            #     images = images.view([-1, 3, 1600, 1600])
            #     # (256,3,1600,1600)
            #
            #     # print(images.shape)
            #
            #     rotate_flags = data['rotate_flag']
            #     # (64,4): [[0,1,2,3],[0,1,2,3],...]
            #     rotate_flags = rotate_flags.view([-1, 1])
            #     # (256,1)
            #
            #     '''
            #     for image, rotate_flag in zip(images, rotate_flags):
            #         pil_image = F.to_pil_image(image)
            #         pil_image.save('/home/std2022/zhaoxu/Document_angle/test1/' + str(index) +
            #                        '_{}'.format(rotate_flag.item()) + '.jpg')
            #
            #         # print('index_{}_flag{}'.format(index,rotate_flag.item()))
            #         i += 1
            #         if i % 4 == 0:
            #             index += 1
            #     '''


    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_DNN(self):

        torch.cuda.empty_cache()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model.train()
        self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)

        Loss, Error = 0,0
        img_num = 0

        i=0

        for data in self.test_loader:
            images = data['image']
            images = images.view([-1,3,1600,1600])
            rotated_flags = data['rotate_flag']
            rotated_flags = rotated_flags.view([-1,1])
            rotated_flags = rotated_flags.squeeze()

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
            loss = self.loss_func(output,rotated_flags)
            # print(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(output)
            pre = output.detach().max(1)[1]
            # print('predict: {}'.format(pre))
            error = self.get_error(pre,rotated_flags)
            Error += error
            Loss += loss.item()

            if i % 10 == 0:
                print("[train] batch: %d, loss: %.3f, error: %.3f" %(i+1,Loss/(i+1),Error/(i+1)))

            i += 1

        # test DNN

        # save model

        # save param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('---model_name', default='resnet50', type=str)
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--device',default='0,1,2',type=str)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # print(args)

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2, 3"

    trainer = ImageStep(args)
    trainer.load_models()

    trainer.load_data(dataset='bank_test')
    trainer.train_DNN()
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.current_device())

if __name__ == '__main__':
    main()

'''
(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
(fc): Linear(in_features=2048, out_features=4, bias=True)
'''
