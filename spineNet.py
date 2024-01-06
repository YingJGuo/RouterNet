from resnet import resnet34
import torch.nn as nn
from decoder import DecNet

class SpineNet(nn.Module):
    def __init__(self, heads,center_num, recurrent_num,local_win,pretrained):
        super(SpineNet, self).__init__()
        channels = [3, 64, 64, 128, 256, 512]
        self.base_network = resnet34(pretrained=pretrained,in_channels=3)
        self.dec_net = DecNet(heads, center_num,recurrent_num,local_win)

    def forward(self, x,batch_size,init_center,init_corner_off,root_idx):
        x = self.base_network(x)
        dec_dict = self.dec_net(x,batch_size,init_center,init_corner_off,root_idx)
        return dec_dict

