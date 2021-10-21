import config.train_config
import dataset.image_dataset
import utils
from torchvision.transforms import functional as F

cfg = config.train_config.TrainConfig()

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

for data in bank_loader['train']:

    images = data['image']
    # print(images.shape)
    # (64,4,3,1600,1600)

    images = images.view([-1,3,1600,1600])
    print(images.shape)
    # (256,3,1600,1600)

    # print(images.shape)
    # （64,3,1600,1600)

    rotate_flags = data['rotate_flag']
    # (64,4): [[0,1,2,3],[0,1,2,3],...]
    rotate_flags = rotate_flags.view([-1,1])
    # (256,1)

    for image, rotate_flag in zip(images, rotate_flags):
        pil_image = F.to_pil_image(image)
        pil_image.save('/home/std2022/zhaoxu/Document_angle/train1/' + str(index) +
                       '_{}'.format(rotate_flag.item()) + '.jpg')

        # print('index_{}_flag{}'.format(index,rotate_flag.item()))
        # i += 1
        # if i % 4 == 0:
        #     index += 1
        #
        # if i > 100:
        #     break

