import config.train_config
import dataset.image_dataset
import utils
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from transform.basic_transform import ScaleResize, Selector
from itertools import cycle
from PIL import Image

cfg = config.train_config.TrainConfig()

# train_loader_bank
# train_loader_doc

bank = {}
doc = {}
bank['train'] = dataset.image_dataset.Bank_Train(cfg)
bank['val'] = dataset.image_dataset.Bank_Val(cfg)
bank['test'] = dataset.image_dataset.Bank_Test(cfg)
doc['train'] = dataset.image_dataset.Doc_Train(cfg)
doc['val'] = dataset.image_dataset.Doc_Val(cfg)
doc['test'] = dataset.image_dataset.Doc_Test(cfg)

# print(bank['train'])
# banktttt = bank['train']
# print(len(banktttt)) # 1168

# print(banktttt[1].shape)
# [1,1400,1400]
# print('\n')

# print(bank['test'])

bank_loader = utils.generate_data_loaders(bank)
doc_loader = utils.generate_data_loaders(doc)

bank_loader_train = bank_loader['train']
doc_loader_train = doc_loader['train']

bank_loader_test = bank_loader['test']
doc_loader_test = doc_loader['test']
# print(bank_loader_train)

# for data in bank_loader_train:
#     print(data.shape)


# print("hello")
#
# index = 0
# i = 0
# k = 0
# # bank & document [train]
#

for i, (bank_data, doc_data) in enumerate(zip(cycle(bank_loader_test), doc_loader_test)):
    # print(bank_data['image'].shape)
    # print(bank_data['rotate_flag'].shape)
    # rotate_flags = bank_data['rotate_flag'].squeeze()
    # print(rotate_flags)

    bank_images = bank_data['image']
    bank_images = bank_images.view([-1, 1, 1400, 1400])
    doc_images = doc_data['image']
    doc_images = doc_images.view([-1, 1, 1400, 1400])
    images = torch.concat((bank_images, doc_images), 0)

    bank_flags = bank_data['rotate_flag']
    bank_flags = bank_flags.view([-1, 1])
    bank_flags = bank_flags.squeeze()
    doc_flags = doc_data['rotate_flag']
    doc_flags = doc_flags.view([-1, 1])
    doc_flags = doc_flags.squeeze()
    # print(bank_flags)
    # print(doc_flags)
    if bank_flags.ndim == 0:
        bank_flags = bank_flags.unsqueeze(0)
    if doc_flags.ndim == 0:
        doc_flags = doc_flags.unsqueeze(0)

    rotated_flags = torch.concat((bank_flags, doc_flags), 0)
    print(bank_flags)
    print(doc_flags)
    print(rotated_flags)
    print(rotated_flags.shape)
    print('---------')
    if rotated_flags.ndim == 0:
        break

# for i, (bank_data, doc_data) in enumerate(zip(cycle(bank_loader_train), doc_loader_train)):
#     # rotate_flags = bank_data['rotate_flag'].squeeze()
#     # print(rotate_flags)
#
#     # bank_images = bank_data['image']
#     # print(bank_images.shape)
#     # bank_images = bank_images.view([-1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE])
#     # doc_images = doc_data['image']
#     # doc_images = doc_images.view([-1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE])
#     # images = torch.concat((bank_images, doc_images), 0)
#
#     bank_flags = bank_data['rotate_flag']
#     bank_flags = bank_flags.view([-1, 1])
#     bank_flags = bank_flags.squeeze()
#     doc_flags = doc_data['rotate_flag']
#     doc_flags = doc_flags.view([-1, 1])
#     doc_flags = doc_flags.squeeze()
#     # print(bank_flags)
#     # print(doc_flags)
#
#     rotated_flags = torch.concat((bank_flags, doc_flags), 0)
#     print(bank_flags)
#     print(bank_flags.shape)
#     print(doc_flags)
#     print(rotated_flags)
#     print(rotated_flags.shape)
#     print('---------')
#     break

'''
image_size = cfg.IMAGE_SIZE[0]
for i, (bank_data,doc_data) in enumerate(zip(bank_loader_train,doc_loader_train)):
    # print(type(data))
    # [8,4,1,1400,1400]
    bank_images = bank_data['image']
    doc_images = doc_data['image']
    bank_images = bank_images.view([-1,1,image_size,image_size])
    doc_images = doc_images.view([-1,1,image_size,image_size])
    print(bank_images.shape)
    print("hello\n")
    # print(doc_images.shape)
    # images = torch.concat((bank_images,doc_images),0)
    # print("good\n")
    # print(images.shape)
    # print(i)
    if i >5:
        break

    # print(bank_data['image'].shape)
print(i)
'''

# load one image
# /home/std2022/zhaoxu/disk/Bank/Images/0047.jpg
#        item_dict['image'] = Image.open(item_dict['image_path']).convert('RGB')
#         data = self.transform(item_dict)
# image = input_dict['image']
# image = self.resize(image)
# image = self.gray_scale(image)
#
# # return the 4 rotated copies of the image and the flag of the rotation
# # i.e. 0 for 0 degrees, 1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees
# image = self.to_tensor(image)
#
# image = Image.open('/home/std2022/zhaoxu/disk/Bank/Images/0047.jpg').convert('RGB')
# transform = transforms.Compose([
#     ScaleResize(fixed_size=cfg.IMAGE_SIZE,
#                 fill_value=(255, 255, 255)),
#     transforms.Grayscale(1),
#     transforms.ToTensor()
# ])
#
# image = transform(image)
# image = torch.unsqueeze(image, 0)
# print(image.shape)
#
# j = 0
# for i in range(5):
#     print(i)
#     j += 1
#
# print(j)

# [1, 1, 1000, 1000]
# image = F.to_pil_image(image)

# concat
# if k > 2:
#     break
#
# for image in data:
#     # print(images.shape)
#     if i > 20:
#         break
#     # pil_image = F.to_pil_image(image)
#     # pil_image.save('/home/std2022/zhaoxu/Document_angle/train_imgs/bank_train1/train3/' + str(k) +
#     #                '_{}'.format(i) + '.jpg')
#     # print(i)
#     # index += 1
#     i += 1
#
# k += 1

# images = data['image']
# # print(images.shape)
# # (64,4,3,1600,1600)
# print(images.shape)
#
# images = images.view([-1,1,1400,1400])
# print(images.shape)
# # (256,3,1600,1600)
#
# # print(images.shape)
# # （64,3,1600,1600)
#
# rotate_flags = data['rotate_flag']
# # (64,4): [[0,1,2,3],[0,1,2,3],...]
# rotate_flags = rotate_flags.view([-1,1])
# # (256,1)

# for image, rotate_flag in zip(images, rotate_flags):
#     if i > 100:
#         break
#     # pil_image = F.to_pil_image(image)
#     # pil_image.save('/home/std2022/zhaoxu/Document_angle/train_imgs/train2/' + str(index) +
#     #                '_{}'.format(rotate_flag.item()) + '.jpg')
#
#     print('index_{}_flag{}'.format(index,rotate_flag.item()))
#     i += 1
#     if i % 4 == 0:
#         index += 1
#
# print("Good {}".format(i))

#
# for data in doc_loader['train']:
#     print(data)
#
#     images = data['image']
#     print(images.shape)
#     # (64,4,3,1600,1600)
#     print(images.shape)
#
#     images = images.view([-1, 1, 1400, 1400])
#     print(images.shape)
#     # (256,3,1600,1600)
#
#     # print(images.shape)
#     # （64,3,1600,1600)
#
#     rotate_flags = data['rotate_flag']
#     # (64,4): [[0,1,2,3],[0,1,2,3],...]
#     rotate_flags = rotate_flags.view([-1, 1])
#     # (256,1)
#
#     for image, rotate_flag in zip(images, rotate_flags):
#         if i > 100:
#             break
#         pil_image = F.to_pil_image(image)
#         pil_image.save('/home/std2022/zhaoxu/Document_angle/train_imgs/doc_train1/' + str(index) +
#                        '_{}'.format(rotate_flag.item()) + '.jpg')
#
#         print('index_{}_flag{}'.format(index, rotate_flag.item()))
#         i += 1
#         if i % 4 == 0:
#             index += 1

# print("Good {}".format(i))


# for data in enumerate(zip(bank_loader['train'],doc_loader['train'])):
#     images = data['image']
#     print(images)
