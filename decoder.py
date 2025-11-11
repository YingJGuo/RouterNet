import torch.nn as nn
import torch.nn.functional as F
import torch
from PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
from einops import rearrange, repeat


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y

class DecNet(nn.Module):
    def __init__(self, heads,center_num,recurrent_num,local_win):
        super(DecNet, self).__init__()
        channel_list = [512,256,128,64,64]
        self.dec_c1 = CombinationModule(64, 64, group_norm=True)#2
        self.dec_c2 = CombinationModule(128, 64, group_norm=True)#4
        self.dec_c3 = CombinationModule(256, 128, group_norm=True)#8
        self.dec_c4 = CombinationModule(512, 256, group_norm=True)#16

        avg_pool_size = 1
        avg_pool_size_vec = 10
        self.average_pool = nn.AdaptiveAvgPool2d(avg_pool_size)
        self.average_pool_vec = nn.AdaptiveAvgPool2d(avg_pool_size_vec)
        self.heads = heads
        self.center_num = center_num
        self.recurrent_num = recurrent_num
        self.local_win_list = local_win
        
        for head in self.heads:
            classes = self.heads[head]
                
            if head == 'center_pt':#2*1
                fc = nn.Linear(channel_list[0]*1*avg_pool_size*avg_pool_size, classes)
                self.fill_fc_weights(fc)
                self.__setattr__(head, fc)
            elif head=='center':#2*17
                fc_center_off_0 = nn.Linear(channel_list[1]*self.center_num*avg_pool_size*avg_pool_size, classes)
                self.fill_fc_weights(fc_center_off_0)
                self.__setattr__(head+'_off_0', fc_center_off_0)

                fc_center_off_1 = nn.Linear(channel_list[2]*self.center_num*avg_pool_size*avg_pool_size, classes)
                self.fill_fc_weights(fc_center_off_1)   
                self.__setattr__(head+'_off_1', fc_center_off_1) 
                
            elif head=='corner':#2*68
                fc_corner_off_0 = nn.Linear(channel_list[3]*avg_pool_size*avg_pool_size*4, classes)
                self.fill_fc_weights(fc_corner_off_0)
                self.__setattr__(head+'_off_0', fc_corner_off_0)
                
                fc_corner_off_1 = nn.Linear(channel_list[4]*avg_pool_size*avg_pool_size*4, classes)
                self.fill_fc_weights(fc_corner_off_1)
                self.__setattr__(head+'_off_1', fc_corner_off_1)
                
            
    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prroi_feat(self,batch_size,input_coord,pt_num,input_features,roi_width,roi_height,down_ratio,is_first=False):
        #get rois
        repeat_shapes = repeat(input_coord,'batch num coord -> batch num (N coord)',num = pt_num, N = 2)#repeat to N,17,4
        reshape_shapes = rearrange(repeat_shapes,'batch num coord -> (batch num) coord',num = pt_num,coord = 4)#reshape to 17N,4
        roi_off_width = roi_width*down_ratio/2
        roi_off_height = roi_height*down_ratio/2
        if is_first:
            offset = torch.Tensor([-roi_off_width,0,roi_off_width,2*roi_off_height]).cuda()
        else:
            offset = torch.Tensor([-roi_off_width,-roi_off_height,roi_off_width,roi_off_height]).cuda()
        bbox = reshape_shapes+offset
        batch_idx = torch.arange(0,batch_size).cuda()
        batch_idx = repeat(batch_idx,'batch -> (batch num)',num = pt_num)#repeat to N,17,4
          
        batch_idx = torch.unsqueeze(batch_idx,1)
        rois = torch.cat((batch_idx,bbox),1)
     
        #get roi features 
        roi_features = prroi_pool2d(input_features, rois, roi_height, roi_width, 1/down_ratio)
        return roi_features

    def classify_layer(self,head,input_feat,center_position,pt_num,batch_size):

        if head=='center_vec':
            roi_feature = self.average_pool_vec(input_feat)
        else:
            roi_feature = self.average_pool(input_feat)
        
        roi_feature = roi_feature.reshape(batch_size, -1)
        offset = self.__getattr__(head)(roi_feature)
        
        offset_cache = rearrange(offset,'batch (num coord) -> batch num coord',num = pt_num,coord = 2)
        new_position = center_position + offset_cache
        return offset,new_position
        

    def forward(self, x,batch_size,init_center,init_corner_off,root_idx):
        feature_list = []
        c5_feat = x[-1] #down 32
        c4_combine_feat = self.dec_c4(x[-1], x[-2])#down 16
        c3_combine_feat = self.dec_c3(c4_combine_feat, x[-3])#down 8
        c2_combine_feat = self.dec_c2(c3_combine_feat, x[-4])#down 4
        c1_combine_feat = self.dec_c1(c2_combine_feat, x[-5])#down 2
        feature_list.append(c5_feat)
        feature_list.append(c4_combine_feat)
        feature_list.append(c3_combine_feat)
        feature_list.append(c2_combine_feat)
        feature_list.append(c1_combine_feat)
        #get multi-scale roi features & offset & positions
        last_offset = 0
        last_position = init_center[:,root_idx,:]
        last_position = torch.unsqueeze(last_position,1)


        dec_dict = {}

        #predicting the root pt
        dec_dict['init_pt_pos'] = last_position
        
        roi_features_0 = self.prroi_feat(batch_size,last_position,1,feature_list[0],\
                (self.local_win_list[0,0]//32),(self.local_win_list[0,1]//32),32)
        offset_0 , position_0 = self.classify_layer('center_pt', roi_features_0 , last_position,1 , batch_size)# b,1,2
        dec_dict['center_pt_pos'] = position_0

        center_pts_pos_0 = offset_0.unsqueeze(1) +init_center
        dec_dict['center_pts_pos_0'] = center_pts_pos_0
        
        roi_features_1 = self.prroi_feat(batch_size, center_pts_pos_0,17,feature_list[1],\
                (self.local_win_list[1,0]//16),(self.local_win_list[1,1]//16),16)
        offset_1 , center_pts_pos_1 = self.classify_layer('center_off_0', roi_features_1 , center_pts_pos_0,17 , batch_size)# b,17,2
        dec_dict['center_pts_pos_1'] = center_pts_pos_1
        
        roi_features_2 = self.prroi_feat(batch_size,center_pts_pos_1,17,feature_list[2],\
                (self.local_win_list[2,0]//8),(self.local_win_list[2,1]//8),8)
        offset_2 , center_pts_pos_2 = self.classify_layer('center_off_1', roi_features_2 , center_pts_pos_1,17 , batch_size)# b,17,2
        dec_dict['center_pts_pos_2'] = center_pts_pos_2# b,17,2
        
        #predicting the corner pts
        center_pts_pos_2 = repeat(center_pts_pos_0,'batch num coord -> batch (num N) coord',num = 17, N = 4)#repeat to N,17,4
        corner_pts_pos_0 = center_pts_pos_2+init_corner_off
        dec_dict['corner_pts_pos_0'] = corner_pts_pos_0 # b,68,2
        
        roi_features_3 = self.prroi_feat(batch_size,corner_pts_pos_0,17*4,feature_list[3],\
                (self.local_win_list[3,0]//4),(self.local_win_list[3,1]//4),4)
        roi_features_3 = self.average_pool(roi_features_3)
        roi_features_3 = roi_features_3.reshape(batch_size*self.center_num,-1)
        corner_off_0 = self.__getattr__('corner_off_0')(roi_features_3)
        corner_off_0 = corner_off_0.reshape(batch_size,68,2)
        corner_pts_pos_1 = corner_pts_pos_0+corner_off_0
        dec_dict['corner_pts_pos_1'] = corner_pts_pos_1# b,68,2

        roi_features_4 = self.prroi_feat(batch_size,corner_pts_pos_1,17*4,feature_list[4],\
                (self.local_win_list[4,0]//2),(self.local_win_list[4,1]//2),2)
        roi_features_4 = self.average_pool(roi_features_4)
        roi_features_4 = roi_features_4.reshape(batch_size*self.center_num,-1)
        corner_off_1 = self.__getattr__('corner_off_1')(roi_features_4)
        corner_off_1 = corner_off_1.reshape(batch_size,68,2)
        corner_pts_pos_2 = corner_pts_pos_1+corner_off_1
        dec_dict['corner_pts_pos_2'] = corner_pts_pos_2# b,68,2
        
        
        '''
        return: a set of features, it can be used for Cal Loss 
        '''
        return dec_dict

class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((x_up, x_low), 1))