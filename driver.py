import config.train_config
import dataset.image_dataset
import utils
import torch
from torchvision.transforms import functional as F

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

