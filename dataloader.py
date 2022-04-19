import numpy as np
import os, glob
import csv
import scipy.io as sio
from torch.utils.data import Dataset


class HS(Dataset):

    def __init__(self, root, mode):

        super(HS, self).__init__()
        self.root = root
        self.mode = mode
        self.names = ['LRHS', 'PAN', 'gtHS']
        if mode == 'train':
            self.path = 'images_train.csv'
        else:
            self.path = 'images_test.csv'
        self.images_L, self.images_P, self.labels = self.load_csv(self.path)

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):

            images_LRHS = sorted(glob.glob(os.path.join(self.root, self.names[0], self.mode, '*.mat')),
                                 key=lambda x: int(x.split('.')[0].split('_')[-1]))
            images_PAN = sorted(glob.glob(os.path.join(self.root, self.names[1], self.mode, '*.mat')),
                                key=lambda x: int(x.split('.')[0].split('_')[-1]))
            images_gtHS = sorted(glob.glob(os.path.join(self.root, self.names[2], self.mode, '*.mat')),
                                 key=lambda x: int(x.split('.')[0].split('_')[-1]))

            print(len(images_LRHS), images_LRHS)
            print(len(images_PAN), images_PAN)
            print(len(images_gtHS), images_gtHS)

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(images_LRHS)):
                    writer.writerow([images_LRHS[i], images_PAN[i], images_gtHS[i]])
                print('writen into csv file:', filename)

        images_L, images_P, labels = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img1, img2, label = row
                images_L.append(img1)
                images_P.append(img2)
                labels.append(label)

        assert len(images_L) == len(labels) and len(images_P) == len(labels)

        return images_L, images_P, labels

    def __len__(self):
        return len(self.images_L)

    def __getitem__(self, item):

        img1, img2, label = self.images_L[item], self.images_P[item], self.labels[item]

        data_img1 = np.squeeze(sio.loadmat(img1)['b'].astype(np.float32))
        data_img2 = sio.loadmat(img2)['b'].astype(np.float32)
        data_img2 = np.reshape(data_img2, [1, 160, 160])
        data_label = np.squeeze(sio.loadmat(label)['b'].astype(np.float32))

        return data_img1, data_img2, data_label
