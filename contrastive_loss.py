# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:37:12 2022

@author: loua2
"""
import torch
from torch import nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class ConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())#两相同的无标签图像激活图，大小为[1, 16, 128, 72]
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)#[1, 9216, 16]，9216个pixels
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)#[1, 9216, 16]，9216个pixels
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))#[9216, 1, 1],batchsize不变的矩阵乘法,相当于对应pixel位置乘，所以是正样本
        l_pos = l_pos.view(-1, 1)#[9216, 1]

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)#[1, 9216, 16]
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)#[1, 9216, 16]
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))#bmm([1, 9216, 16],[1, 16, 9216])
        #得出project1输出的q和project1输出的k的对应所有位置的关系
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)#把那些同一个位置的数值设置为-10，排除自相关的影响
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature
        #第一列表示所有相同位置之间的相关性，要他大！
        #从第二列开始表示所有位置和其他所有位置的相关性，要他小！

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))#这是对的，因为全0相当于第一类，也就是onehot的第一个元素为1，所以该loss可以达到上面说的。。。

        return loss
    
    
class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())#[1, 32, 64, 36]
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)#两个不同的图像[1, 2304, 32]
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)#两个不同的图像[1, 2304, 32]
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.zeros((batch_size*feat_q.shape[1],1)).cuda()
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss  
    
  
    

class ConLoss_class(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss_class, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q,mask_l):
        ##正相关的位置：
        neg_f=[]
        l_pos=[]
        for i in range(4):
            feat_q1 = F.normalize(feat_q[i], dim=-1, p=1)
            neg_f.append(feat_q1)
            # pos logit
            l_pos.append(torch.bmm(feat_q1, feat_q1.transpose(2, 1)))#表示所有正样本关系值，我们希望他大


        #求负相关的：
        loss_fn = nn.BCEWithLogitsLoss()
        loss=0.


        for ii in range(4):#对第ii个类别求他和其他类别的关系
            if torch.sum(mask_l==ii+1)==1 or  torch.sum(mask_l==ii+1) < 0.1:  # 当这个类别一个特征都没有那就没有相关和无关向量
                continue
            for i in range(4):#对这第ii个类别求他和其他第i哥类别的关系
                if torch.sum(mask_l==i+1)==1 or  torch.sum(mask_l==i+1) < 0.1:#当这个类别没有
                    continue
                if i == ii:#不求自己，求自己相当于正相关了
                    continue
                l_neg = torch.bmm(neg_f[ii], neg_f[i].transpose(2, 1))
                l_pos[ii] = torch.cat((l_pos[ii], l_neg), dim=2)
            l_pos[ii]= l_pos[ii].squeeze()
            oones=torch.ones((l_pos[ii].shape[0],l_pos[ii].shape[0]), dtype=torch.long,
                                                            device=mask_l.device)
            try:
                # 你的代码逻辑
                zzeros = torch.zeros((l_pos[ii].shape[0], l_pos[ii].shape[1] - l_pos[ii].shape[0]), dtype=torch.float,
                                     device=mask_l.device)
            except Exception as e:
                print(f"发生错误：{e}")

            loss+=loss_fn(l_pos[ii],torch.cat((oones, zzeros), dim=1))



        return loss