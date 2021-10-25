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
import config.train_config
import dataset.image_dataset
import utils.util_data
from itertools import cycle
import argparse
import time
import warnings

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

# checkpoint --> epoch

cfg = config.train_config.TrainConfig()
warnings.filterwarnings("ignore")
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
        self.path['root'] = args.root
        self.path['result_path'] = os.path.join(self.path['root'],args.result_dir)
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
        self.batch_size = args.batch_size
        self.list = [[],[],[],[]]
        self.save_hparam(args)
        self.train_test = args.train_test
        self.device_ids = [0,1,2]

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

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model,device_ids=self.device_ids)

        if self.start_epoch != 0:
            self.load_latest_epoch()

        self.loss_func = nn.CrossEntropyLoss()

        if self.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load_trained_model(self):
        # original saved file with DataParallel
        model_path = self.path['result_path']+ self.model_name+'_model_G'+str(self.image_size)+'_'+str(self.epoch_num-1)+'.pkl'
        loaded_dict = torch.load(model_path)
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False, num_classes=self.num_class)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model.state_dict = loaded_dict

    def load_latest_epoch(self):
        model_path = self.path['result_path']+self.model_name+'_model_G'+str(self.image_size)+'_'+str(self.start_epoch-1)+'.pkl'
        list_path = self.path['result_path']+self.model_name+'_list_G'+str(self.image_size)+'_'+str(self.start_epoch-1)+'.bin'
        loaded_dict = torch.load(model_path)
        self.model.state_dict = loaded_dict
        self.list = torch.load(list_path)

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
        self.model = self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)

        Loss, Error = 0, 0
        # img_num = 0

        for i,(bank_data,doc_data) in enumerate(zip(cycle(self.train_loader_bank),self.train_loader_doc)):
            # 4*4*2
            bank_images = bank_data['image']
            bank_images = bank_images.view([-1, 1, self.image_size, self.image_size])
            doc_images = doc_data['image']
            doc_images = doc_images.view([-1,1,self.image_size,self.image_size])
            images = torch.concat((bank_images,doc_images),0)
            # print(images.shape)
            # If batch_size == 4, the shape of input data is: [32,1,1300,1300],
            # where the  32 means 4 batch_size with 4 rotation angles --> 16
            # band and doc data --> 16 * 2 == 32
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
        self.list[0].append(Loss/(i+1))
        self.list[1].append(Error/(i+1))

    def val_DNN(self):
        self.model.eval()
        self.model.to(self.device)
        Error = 0
        with torch.no_grad():
            for i, (bank_data,doc_data) in enumerate(zip(cycle(self.val_loader_doc),self.val_loader_bank)):
                bank_images = bank_data['image']
                bank_images = bank_images.view([-1, 1, self.image_size, self.image_size])
                doc_images = doc_data['image']
                doc_images = doc_images.view([-1,1,self.image_size,self.image_size])
                images = torch.concat((bank_images,doc_images),0)

                bank_flags = bank_data['rotate_flag']
                bank_flags = bank_flags.view([-1,1])
                bank_flags = bank_flags.squeeze()
                doc_flags = doc_data['rotate_flag']
                doc_flags = doc_flags.view([-1,1])
                doc_flags = doc_flags.squeeze()
                rotated_flags = torch.concat((bank_flags,doc_flags),0)

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
            self.list[2].append(Error/(i+1))

    # test DNN
    def test_DNN(self):

        self.model.eval()
        self.model.to(self.device)

        Error = 0
        with torch.no_grad():
            for i, (bank_data,doc_data) in enumerate(zip(cycle(self.test_loader_bank),self.test_loader_doc)):
                # because test_doc is longer than test_bank
                bank_images = bank_data['image']
                bank_images = bank_images.view([-1, 1, self.image_size, self.image_size])
                doc_images = doc_data['image']
                doc_images = doc_images.view([-1,1,self.image_size,self.image_size])
                images = torch.concat((bank_images,doc_images),0)

                bank_flags = bank_data['rotate_flag']
                bank_flags = bank_flags.view([-1,1])
                bank_flags = bank_flags.squeeze()
                doc_flags = doc_data['rotate_flag']
                doc_flags = doc_flags.view([-1,1])
                doc_flags = doc_flags.squeeze()
                rotated_flags = torch.concat((bank_flags,doc_flags),0)

                images = images.float()
                images = images.to(self.device)
                rotated_flags = rotated_flags.to(self.device)
                output = self.model(images)
                # loss = self.loss_func(output, rotated_flags)
                pre = output.detach().max(1)[1]
                error = self.get_error(pre, rotated_flags)

                Error += error
                if i % 10 == 0:
                    print("[test] batch: %d, error: %.3f" % (i + 1, Error / (i + 1)))

            self.list[3].append(Error / (i + 1))
            test_error = self.list[3]
            print('Error on test dataset: {}'.format(test_error[0]))
            self.save_test()
            # break


    def work_train(self, args):
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epoch_num)
        for epoch in range(self.start_epoch, self.epoch_num):
            print("epoch %d:" % (epoch+1))
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
            time_remain = time_expand*(self.epoch_num-epoch-1)
            time_scale = 's'
            if time_remain > 60:
                time_scale = 'min'
                time_remain = time_remain/60
            print("Approximate reamaining time: [{}] {}\n".format(time_remain,time_scale))

        # draw figures and save model
        self.draw_figures()
        # self.save_model()
        self.save_list_val()
        # self.save_hparam()

    def draw_figures(self):
        x = np.arange(0,len(self.list[0]),1)
        train_l,train_e,val_e = np.array(self.list[0]),np.array(self.list[1]),\
                                        np.array(self.list[2])
        plt.figure()
        plt.subplot(211)
        plt.plot(x,train_l,color = 'red')
        # plt.plot(x,val_l,color='blue')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(212)
        plt.plot(x,train_e,color='red')
        plt.plot(x,val_e,color='blue')
        plt.xlabel('epoch')
        plt.ylabel('error')

        save_path = self.path['result_path']+self.model_name+'_G'+str(self.image_size)+'_E'+str(self.epoch_num)+'_curve.png'
        plt.savefig(save_path)
        plt.close()

    def save_latest_epoch(self, epoch):
        model_path = self.path['result_path'] + self.model_name +'_model_G'+str(self.image_size) +'_'+ str(epoch) + '.pkl'
        list_path = self.path['result_path'] + self.model_name +'_list_G'+str(self.image_size) + '_'+ str(epoch) + '.bin'
        torch.save(self.model.state_dict, model_path)
        torch.save(self.list, list_path)
        print("Successfully save model {}, epoch {}!".format(self.model_name,epoch))

    # def save_model(self):
    #     save_path = self.path['result_path']+'models/'+self.model_name+'_'+str(self.epoch_num)+'_'+str(self.lr)+'.pkl'
    #     torch.save(self.model.state_dict(),save_path)

    # save hyper_param
    def save_hparam(self,args):
        save_path = self.path['result_path']+self.model_name+'_G'+str(self.image_size)+'_E'+str(self.epoch_num)+'_hparam.txt'
        with open(save_path,'w') as fp:
            args_dict = args.__dict__
            for key in args_dict:
                fp.write("{} : {}\n".format(key,args_dict[key]))

    def save_list_val(self):
        save_path = self.path['result_path']+self.model_name+'_loss_G'+str(self.image_size)+'_Ep'+str(self.epoch_num)+'_train_val.txt'
        with open(save_path,'w') as fp:
            train_loss = self.list[0]
            train_error = self.list[1]
            val_error = self.list[2]

            for i in range(len(train_loss)):
                fp.write('epoch:{}, train_loss:{}, train_error:{}, val_error:{}\n'.format(i+1,
                        train_loss[i],train_error[i],val_error[i]))
            print("Successfully save train val lists!")

    def save_test(self):
        save_path = self.path['result_path']+self.model_name+'_G'+str(self.image_size)+'_E'+str(self.epoch_num)+'_test.txt'
        save_path = os.path.join(save_path)
        with open(save_path,'w') as fp:
            test_error = self.list[3]
            fp.write('test error:{}\n'.format(test_error[0]))
        print("Successfully save test error!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='/home/std2022/zhaoxu/')
    parser.add_argument('--result-dir',default='Document_angle/result/result1/')
    parser.add_argument('--train-test',default='train',type=str)
    parser.add_argument('--num-class', default=4, type=int)
    parser.add_argument('--model-name', default='resnet34', type=str)
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--logspace', default=1, type=float)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--epoch-num', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--image-size',default=cfg.IMAGE_SIZE[0],type=int)
    parser.add_argument('--batch-size',default=cfg.BATCH_SIZE,type=int)
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--gpu-id',default='1,2,3',type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    seed_num = args.epoch_num + args.seed
    seed_torch(seed_num)

    trainer = ImageStep(args)

    if args.train_test == 'train':
        trainer.load_models()
        trainer.load_data(dataset='bank_doc_train')
        trainer.work_train(args)

    elif args.train_test == 'test':
        trainer.load_trained_model()
        trainer.load_data(dataset='bank_doc_train')
        trainer.test_DNN()


if __name__ == '__main__':
    main()

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

# predict time
# save model and params
# test model

