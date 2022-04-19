import torch
from torch import nn, optim, autograd
from Attentions import SpatialAttention, ChannelAttention


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
    )
    return blk


class RHDB(nn.Module):
    def __init__(self, in_c_1, in_c_2, gr_1, gr_2, out_c_1, out_c_2, num_convs=2):
        super(RHDB, self).__init__()

        self.num_convs = num_convs
        net_up = []
        net_down = []
        for i in range(num_convs):
            if i > 0:
                in_up_c = in_c_1 + i * gr_1 + in_c_2 + (i - 1) * gr_2
                in_down_c = in_c_2 + i * gr_2 + in_c_1 + (i - 1) * gr_1
            else:
                in_up_c = in_c_1
                in_down_c = in_c_2
            net_up.append(conv_block(in_up_c, gr_1))
            net_down.append(conv_block(in_down_c, gr_2))
        self.net_up = nn.ModuleList(net_up)
        self.net_down = nn.ModuleList(net_down)

        ca_input1 = []
        sa_input2 = []
        for _ in range(num_convs):
            ca_input1.append(ChannelAttention(in_c_1))
            sa_input2.append(SpatialAttention())
        self.ca_input1 = nn.ModuleList(ca_input1)
        self.sa_input2 = nn.ModuleList(sa_input2)

        ca_conv1 = []
        sa_conv1 = []
        for _ in range(num_convs - 1):
            ca_conv1.append(ChannelAttention(gr_1))
            sa_conv1.append(SpatialAttention())
        self.ca_conv1 = nn.ModuleList(ca_conv1)
        self.sa_conv1 = nn.ModuleList(sa_conv1)

        self.transition_up = conv_block(in_c_1 + num_convs * gr_1 + in_c_2 + (num_convs - 1) * gr_2, out_c_1)
        self.transition_down = conv_block(in_c_2 + num_convs * gr_2 + in_c_1 + (num_convs - 1) * gr_1, out_c_2)

    def forward(self, input1, input2):

        input1_ca, input2_sa = [], []
        for i in range(self.num_convs):
            input1_ca.append(self.ca_input1[i](input1))
            input2_sa.append(self.sa_input2[i](input2))

        conv1_up, conv1_down = self.net_up[0](input1), self.net_down[0](input2)
        conv1_ca, conv1_sa = [], []
        for i in range(self.num_convs - 1):
            conv1_ca.append(self.ca_conv1[i](conv1_up))
            conv1_sa.append(self.sa_conv1[i](conv1_down))

        conv2_up, conv2_down = self.net_up[1](torch.cat([input1, conv1_up, input2_sa[0]], 1)), self.net_down[1](
            torch.cat([input2, conv1_down, input1_ca[0]], 1))

        out1 = self.transition_up(torch.cat([input1, conv1_up, conv2_up, input2_sa[1], conv1_sa[0]], 1))
        out2 = self.transition_down(torch.cat([input2, conv1_down, conv2_down, input1_ca[1], conv1_ca[0]], 1))

        out1 = torch.add(input1, out1)
        out2 = torch.add(input2, out2)
        return out1, out2


class RHDN(nn.Module):

    def __init__(self, in_c_1, in_c_2, RHDB_num1, RHDB_num2, G1=64, G2=64, gr_1=16, gr_2=16):

        super(RHDN, self).__init__()

        self.vl_ch = in_c_1
        self.RHDB_num1 = RHDB_num1
        self.RHDB_num2 = RHDB_num2
        # SFEN definition of Visible Light Branch
        self.conv_vl_1 = nn.Sequential(
            transition_block(in_c_1, G1),
            transition_block(G1, G1)
        )

        self.conv_PAN_vl_1 = nn.Sequential(
            conv_block(1, G1),
            conv_block(G1, G1)
        )

        # SFEN Definition of Near Infrared Branch
        self.conv_ni_1 = nn.Sequential(
            transition_block(in_c_2, G2),
            transition_block(G2, G2)
        )

        self.conv_PAN_ni_1 = nn.Sequential(
            conv_block(1, G2),
            conv_block(G2, G2)
        )

        # DFEN definition of Visible Light Branch
        net1 = []
        for _ in range(RHDB_num1):
            net1.append(RHDB(G1, G1, gr_1, gr_1, G1, G1))
        self.net1 = nn.ModuleList(net1)

        # DFEN Definition of Near Infrared Branch
        net2 = []
        for _ in range(RHDB_num2):
            net2.append(RHDB(G2, G2, gr_2, gr_2, G2, G2))
        self.net2 = nn.ModuleList(net2)

        # FFN definition of Visible Light Branch and Near Infrared Branch
        self.conv_vl = conv_block(RHDB_num1 * G1 * 2, RHDB_num1 * G1 * 2)
        self.conv_ni = conv_block(RHDB_num2 * G2 * 2, RHDB_num2 * G2 * 2)

        # DRN definition
        self.conv1 = transition_block(RHDB_num2 * G2 * 2 + RHDB_num1 * G1 * 2, 128)
        self.conv2 = transition_block(128, 128)
        self.transition = nn.Conv2d(128, in_c_1 + in_c_2, kernel_size=1, stride=1)

    def forward(self, input1, input2):

        input1_vl = input1[:, :self.vl_ch, :, :]
        input1_ni = input1[:, self.vl_ch:, :, :]

        # Visible Light Branch
        # SFEN Forward Propagation
        out_vl = self.conv_vl_1(input1_vl)
        out_pan_vl = self.conv_PAN_vl_1(input2)

        # DFEN Forward Propagation
        out_list_vl = []
        out_list_pan_vl = []
        for rhdb in self.net1:
            out_vl, out_pan_vl = rhdb(out_vl, out_pan_vl)
            out_list_vl.append(out_vl)
            out_list_pan_vl.append(out_pan_vl)

        out_vl = out_list_vl[0]
        out_pan_vl = out_list_pan_vl[0]
        for i in range(self.RHDB_num1 - 1):
            out_vl = torch.cat([out_vl, out_list_vl[i + 1]], 1)
            out_pan_vl = torch.cat([out_pan_vl, out_list_pan_vl[i + 1]], 1)

        # FFN Forward Propagation
        out1 = torch.cat([out_vl, out_pan_vl], 1)
        out1 = self.conv_vl(out1)

        # Near Infrared Branch
        # SFEN Forward Propagation
        out_ni = self.conv_ni_1(input1_ni)
        out_pan_ni = self.conv_PAN_ni_1(input2)

        # DFEN Forward Propagation
        out_list_ni = []
        out_list_pan_ni = []
        for rhdb in self.net2:
            out_ni, out_pan_ni = rhdb(out_ni, out_pan_ni)
            out_list_ni.append(out_ni)
            out_list_pan_ni.append(out_pan_ni)

        out_ni = out_list_ni[0]
        out_pan_ni = out_list_pan_ni[0]
        for i in range(self.RHDB_num2 - 1):
            out_ni = torch.cat([out_ni, out_list_ni[i + 1]], 1)
            out_pan_ni = torch.cat([out_pan_ni, out_list_pan_ni[i + 1]], 1)

        # FFN Forward Propagation
        out2 = torch.cat([out_ni, out_pan_ni], 1)
        out2 = self.conv_ni(out2)

        # DRN Forward Propagation
        out = torch.cat([out1, out2], 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.transition(out)
        out = torch.add(out, input1)

        return out
