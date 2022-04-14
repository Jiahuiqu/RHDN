import torch
import numpy as np
import random
from torch.nn import functional as F
from torch import nn, optim, autograd
from Model import RHDN
from dataloader import HS
from torch.utils.data import DataLoader
import os
import scipy.io as sio

# set random seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda')
epoches = 500
batchsz = 4
vl_num = 79  # Number of Visible Light Bands
ni_num = 23  # Number of Near Infrared Bands
learning_rate = 0.001
RHDB_num1 = 2  # Number of RHDBs in Visible Light branch
RHDB_num2 = 4  # Number of RHDBs in Near Infrared branch

data_path = ''  # data load path
weights_path = ''  # Weights saving path
fusion_path = ''  # Fusion result save path

# Data Load
train_db = HS(data_path, mode='train')
test_db = HS(data_path, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=10, drop_last=False)
test_loader = DataLoader(test_db, batch_size=1, num_workers=10)


def main():
    Train = True
    Test = False

    model = RHDN(vl_num, ni_num, RHDB_num1=RHDB_num1, RHDB_num2=RHDB_num2)
    model.to(device)
    print(model)

    optim_model = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_model, milestones=[150, 300, 450], gamma=0.1, last_epoch=-1)
    best_loss = 1
    best_epoch = -1
    loss_fn_l1 = nn.L1Loss()

    if Train:
        model.train()
        for epoch in range(epoches):
            for step, (LRHS, PAN, gtHS) in enumerate(train_loader):

                LRHS = F.interpolate(LRHS, mode='bilinear', scale_factor=4)
                LRHS = LRHS.type(torch.float).to(device)
                PAN = PAN.type(torch.float).to(device)
                gtHS = gtHS.type(torch.float).to(device)

                optim_model.zero_grad()
                HS_fusion = model(LRHS, PAN)
                loss = loss_fn_l1(HS_fusion, gtHS)
                loss.backward()
                optim_model.step()

                if best_loss > loss:
                    best_loss = loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), weights_path + '/net_weights.pth')

            scheduler.step()
            print("epoch:", epoch, "loss:", loss.item())
        print("best_loss:", best_loss, "best_epoch:", best_epoch)

    if Test:
        model.load_state_dict(torch.load(weights_path + '/net_weights.pth'))
        model.eval()
        index_test = 1
        with torch.no_grad():
            for step, (LRHS, PAN, _) in enumerate(test_loader):
                LRHS = F.interpolate(LRHS, mode='bilinear', scale_factor=4)
                LRHS = LRHS.type(torch.float).to(device)
                PAN = PAN.type(torch.float).to(device)

                HS_fusion = model(LRHS, PAN)
                HS_fusion = np.array(HS_fusion.cpu())

                path = os.path.join(fusion_path, 'fusion_' + str(index_test) + '.mat')

                index_test = index_test + 1
                sio.savemat(path, {'da': HS_fusion.squeeze()})

            print('test finished!')


if __name__ == '__main__':
    main()
