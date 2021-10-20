import os
import os.path as path
import pandas as pd

src_path = '../disk'
trgt_path = './Data'
os.makedirs(trgt_path, exist_ok=True)


def make_Bank():
    dataset_path = path.join(src_path, 'Bank')

    def build(gt_file, csv_file):
        with open(path.join(dataset_path, gt_file), 'r') as f:
            info = f.read()
            lines = info.split('\n')
            lines = [line.split('\t') for line in lines]

        items = []

        for line in lines:
            if line == ['']:
                continue

            info_dict = {}
            info_dict['image_path'] = path.abspath(
                path.join(dataset_path, line[0]))
            items.append(info_dict)

        df = pd.DataFrame(columns=['image_path'])
        df = df.append(items, ignore_index=True)
        df.to_csv(path.join(trgt_path, csv_file), index=False)

    build('GroundTruth_Train.txt', 'Bank_Train.csv')
    build('GroundTruth_Val.txt', 'Bank_Val.csv')
    build('GroundTruth_Test.txt', 'Bank_Test.csv')


def make_Doc():
    dataset_path = path.join(src_path, 'Doc')

    def build(gt_file, csv_file):
        with open(path.join(dataset_path, gt_file), 'r') as f:
            info = f.read()
            lines = info.split('\n')
            lines = [line.split('\t') for line in lines]

        items = []

        for line in lines:
            if line == ['']:
                continue

            info_dict = {}
            info_dict['image_path'] = path.abspath(
                path.join(dataset_path, line[0]))
            items.append(info_dict)

        df = pd.DataFrame(columns=['image_path'])
        df = df.append(items, ignore_index=True)
        df.to_csv(path.join(trgt_path, csv_file), index=False)

    build('GroundTruth_Train.txt', 'Doc_Train.csv')
    build('GroundTruth_Val.txt', 'Doc_Val.csv')
    build('GroundTruth_Test.txt', 'Doc_Test.csv')


if __name__ == '__main__':
    make_Bank()
    make_Doc()
