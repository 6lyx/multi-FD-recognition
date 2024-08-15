import torch
import torchvision.models as models



import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
import copy

from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from dataprocess import data_labeling as dl
device = torch.device('cuda:0')
from PIL import Image
from data_aug_transorm import transform_images
from datetime import datetime
gdx

from loss_semi import loss_sup, loss_diff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, roc_auc_score



import torch.nn as nn
from models.vit_mv import myViT, Transformer_nofu,Transformer_multifuse6,Transformer_multifuse6_self
from models.vit_pytorch.deepvit import DeepViT
from einops import rearrange, repeat
import torch
from einops.layers.torch import Rearrange
import torch.nn.functional as F
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class patchize(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, channels = 3,emb_dropout):
        super(patchize, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent = nn.Identity()

        self.dropout = nn.Dropout(emb_dropout)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x
class patch_senet(nn.Module):
    def __init__(self):
        super(patch_senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(50, 50 // 10),
            nn.ReLU(inplace=True),
            nn.Linear(50 // 10, 50),
            nn.Sigmoid()
        )
    def forward(self, ori_patchs,patchs):
        squeeze = self.avg_pool(ori_patchs).squeeze(2)
        excitation = self.fc(squeeze).unsqueeze(2)
        patchs = excitation.expand_as(ori_patchs) * patchs
        return patchs
class vit_multire(nn.Module):
    def __init__(self, num_classes,dim,depth,heads,mlp_dim, dropout=0.2):
        super(vit_multire, self).__init__()
        #print(x[0].shape,x[1].shape,x[2].shape)
# torch.Size([16, 64, 224, 224]) torch.Size([16, 128, 112, 112]) torch.Size([16, 256, 56, 56])
        self.patchize0 = patchize(
            image_size=224,
            patch_size=32,
            dim=dim,
            channels=3,
            emb_dropout=dropout
        )
        self.patchize1=patchize(
            image_size=112,
            patch_size=16,
            dim=dim,
            channels=64,
            emb_dropout=dropout
        )
        self.patchize2 = patchize(
            image_size=56,
            patch_size=8,
            dim=dim,
            channels=128,
            emb_dropout=dropout
        )
        self.patchize3 = patchize(
            image_size=28,
            patch_size=4,
            dim=dim,
            channels=256,
            emb_dropout=dropout
        )

        self.dim_head=64
        # self.transformer=Transformer_nofu(dim, depth, heads, dim_head= self.dim_head, mlp_dim=mlp_dim, dropout = 0.)
        self.transformer_multifuse=Transformer_multifuse6(dim, depth, heads, dim_head= self.dim_head, mlp_dim=mlp_dim, dropout = 0.)
        self.transformer_multifuse_self = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
                                                                      mlp_dim=mlp_dim,
                                                                      dropout=0.)
        self.transformer_multifuse1 = Transformer_multifuse6(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
                                                            dropout=0.)
        self.transformer_multifuse_self1 = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
                                                                      mlp_dim=mlp_dim,
                                                                      dropout=0.)
        self.transformer_multifuse2 = Transformer_multifuse6(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
                                                            dropout=0.)
        self.transformer_multifuse_self2 = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
                                                                      mlp_dim=mlp_dim,
                                                                      dropout=0.)
        self.transformer_multifuse3 = Transformer_multifuse6(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
                                                            dropout=0.)
        self.transformer_multifuse_self3 = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
                                                                      mlp_dim=mlp_dim,
                                                                      dropout=0.)
        self.transformer_multifuse4 = Transformer_multifuse6(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
                                                             dropout=0.)
        self.transformer_multifuse_self4 = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
                                                                       mlp_dim=mlp_dim,
                                                                       dropout=0.)
        self.transformer_multifuse5 = Transformer_multifuse6(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
                                                             dropout=0.)
        self.transformer_multifuse_self5 = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
                                                                       mlp_dim=mlp_dim,
                                                                       dropout=0.)
        # self.transformer_multifuse_self = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
        #                                                     dropout=0.)
        # self.transformer_multifuse1 = Transformer_multifuse6(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim,
        #                                                     dropout=0.)
        # self.transformer_multifuse_self1 = Transformer_multifuse6_self(dim, depth, heads, dim_head=self.dim_head,
        #                                                               mlp_dim=mlp_dim,
        #                                                               dropout=0.)
        # self.transformer_fuse = Transformer_nofu(dim, depth, heads, dim_head=self.dim_head, mlp_dim=mlp_dim, dropout=0.)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        self.vgg_map = nn.Sequential(
            nn.Linear(25088, 1024),
        )
        self.pool='cls'
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # self.senet0 = patch_senet()
        # self.senet01 = patch_senet()
        # self.senet1 = patch_senet()
        # self.senet11 = patch_senet()
        # self.senet2 = patch_senet()
        # self.senet21 = patch_senet()
        # self.senet3=patch_senet()
        # self.senet31 = patch_senet()
    def forward(self, img,x):
        #先尝试4个patchize，后面尝试一个
        ori_patchs0 = self.patchize0(img)
        ori_patchs1 = self.patchize1(x[0])
        ori_patchs2 = self.patchize2(x[1])
        ori_patchs3 = self.patchize3(x[2])
        #x[4]是最终的VGGactivation map

        # tr_patchs1 = self.transformer(patchs1)
        # tr_patchs2 = self.transformer(patchs2)
        # tr_patchs3 = self.transformer(patchs3)
        # tr_patchs4 = self.transformer(patchs4)
        # tr_patchs5 = self.transformer(patchs5)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse(ori_patchs0, ori_patchs1, ori_patchs2,
                                                                        ori_patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse1(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self1(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse2(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self2(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse3(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self3(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse4(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self4(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse5(patchs0, patchs1, patchs2, patchs3)
        patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self5(patchs0, patchs1, patchs2, patchs3)
        # ori_patchs0 = self.senet0(ori_patchs0, patchs0)
        # ori_patchs1 = self.senet1(ori_patchs1, patchs1)
        # ori_patchs2 = self.senet2(ori_patchs2, patchs2)
        # ori_patchs3 = self.senet3(ori_patchs3, patchs3)

        #patchs0, patchs1, patchs2, patchs3 = self.transformer_multifuse_self(patchs0, patchs1, patchs2, patchs3)
        # patchs0 = self.senet01(ori_patchs0, patchs0)
        # patchs1 = self.senet11(ori_patchs1, patchs1)
        # patchs2 = self.senet21(ori_patchs2, patchs2)
        # patchs3 = self.senet31(ori_patchs3, patchs3)



        # fu_patch = self.transformer_fuse(patchs3)
        # fu_patch = fu_patch.mean(dim=1) if self.pool == 'mean' else fu_patch[:, 0]
        # fu_patch = self.to_latent(fu_patch)
        vgg_out=self.vgg_map(self.adaptive_avg_pool(x[4]).view(x[4].shape[0],-1))

        fu_patch = torch.max(torch.concat((
                                           patchs3[:, 0].unsqueeze(1),vgg_out.unsqueeze(1)), 1), 1)[0]
        patchs4=[]
        patchs5=[]
        output=self.classifier(self.mlp_head(fu_patch.squeeze()))
        return output
def evaluate_multilabel_classification(predictions, labels, threshold=0.5):
    """
    Evaluate multi-label classification model performance.

    :param predictions: Model output probabilities (numpy array of shape (batchsize, num_classes))
    :param labels: Ground truth labels (numpy array of shape (batchsize, num_classes))
    :param threshold: Threshold to convert probabilities into binary outputs
    :return: Dictionary containing evaluation metrics
    """
    # Convert probabilities to binary outputs based on the threshold
    predicted_classes = (predictions >= threshold).astype(int)

    predicted_classes_total=predicted_classes.flatten()
    labels_total=labels.flatten()
    precision_total, recall_total, f1_score_total, _ = precision_recall_fscore_support(labels_total, predicted_classes_total, average=None)
    acc_total=np.mean(predicted_classes == labels)


    # Compute Accuracy Rate for each class
    accuracy_per_class = np.mean(predicted_classes == labels, axis=0)

    # Compute Precision, Recall, and F1-score for each class
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predicted_classes, average=None)

    # Compute Kappa score
    kappa_score = cohen_kappa_score(labels.flatten(), predicted_classes.flatten())
    print('total_f1=',f1_score_total[0],'total_acc=',acc_total,'ks=',kappa_score)
    # Compute AUC for each class and average AUC
    auc = roc_auc_score(labels, predictions, average=None)
    average_auc = np.mean(auc)

    # Compile all the scores into a dictionary
    scores = {
        "Accuracy Rate per class": accuracy_per_class,
        "Precision per class": precision,
        "Recall per class": recall,
        "F1-score per class": f1_score,
        "Kappa score": kappa_score,
        "AUC per class": auc,
        "Average AUC": average_auc
    }
    print(scores)
    return scores

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights

def get_outcomes(probabilities):
    outcomes = {}
    for condition, probability in probabilities.items():
        # Generate a random number between 0 and 1 and compare with the condition's probability
        # If the random number is less than the probability, outcome is 1; otherwise, it's 0
        outcomes[condition] = int(np.random.rand() < probability)
    return outcomes


# Define a new model that outputs the features from each conv block
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGFeatureExtractor, self).__init__()
        self.features = vgg_model.features

    def forward(self, x):
        outputs = []
        for layer in self.features:

            x = layer(x)
            if isinstance(layer, torch.nn.MaxPool2d):
                outputs.append(x)

        return outputs




class VIT_VIT(nn.Module):
    def __init__(self,  num_vit_feature=1000, dim=1024, depth=1, heads=8, mlp_dim=2048,
                 dropout=0.):
        super(VIT_VIT, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(models.vgg19(pretrained=True))
        self.teacher_feature_extractor = VGGFeatureExtractor(models.vgg19(pretrained=True))
        self.image_size=224
        self.patch_size=16
        self.channels=3
        self.vit_multire = vit_multire( num_classes=6,
                                         dim=dim,
                                         depth=depth, heads=heads, mlp_dim=mlp_dim,  dropout=dropout)
        self.avg_pool_layer = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.maskout=torch.zeros((3,224,224))
    def forward(self, img):
        Vgg_output_layers=self.feature_extractor(img)
        #以下是使用数据混合，然后做混合数据的预测一致性监督
        # mask = torch.zeros((img.shape[0]//2, 32 * 32, 49)).to(img.device)
        # num_patchs_to_mask = torch.randint(500, (1,)).item()
        # mask_indx = torch.randint(0, 1024, size=(num_patchs_to_mask,))
        # mask[:, mask_indx, :] = self.alpha
        # mask_img = Rearrange('b (q1 q2) (p1 p2) -> b (q1 p2) (q2 p1)', p1=7, p2=7, q1=32, q2=32)(mask).unsqueeze(1)
        # if img.shape[0] % 2 == 1:
        #     un_label_img =mask_img*img[:-1][::2]+(1-mask_img)*img[1:][::2]
        # else:
        #     un_label_img = mask_img * img[::2] + (1 - mask_img) * img[1:][::2]
        # feat_1 = self.feature_extractor(un_label_img)
        # feat_1 = torch.sigmoid(feat_1[4])
        # with torch.no_grad():
        #     feat_2 = self.teacher_feature_extractor(un_label_img)
        #     feat_2 = torch.sigmoid(feat_2[4])

        #以下是做原数据遮挡，然后做遮挡后的数据还原：
        patch_size=16
        num_patchs=int(224/16*224/16)
        mask = torch.ones((img.shape[0],  num_patchs, patch_size * patch_size)).to(img.device)
        # num_patchs_to_mask = torch.randint(num_patchs//2, (1,)).item()
        num_patchs_to_mask = num_patchs // 2
        mask_indx = torch.randint(0, num_patchs, size=(num_patchs_to_mask,))
        mask[:, mask_indx, :] = 0
        mask_img = Rearrange('b (q1 q2) (p1 p2) -> b (q1 p2) (q2 p1)', p1=patch_size, p2=patch_size, q1=int(224/patch_size), q2=int(224/patch_size))(mask).unsqueeze(1)
        img_m=mask_img*img

        feat_33 = self.feature_extractor(img_m)
        feat_3 =feat_33[4]
        #feat_3 = torch.sigmoid(feat_3[4])
        with torch.no_grad():
            feat_4 = self.teacher_feature_extractor(img)
            feat_4 = feat_4[4]
            #feat_4 = torch.sigmoid(feat_4[4])

        ViT_output=self.vit_multire(img,Vgg_output_layers)
        ViT_output=torch.sigmoid(ViT_output)
        mask_ViT_output = self.vit_multire(img, feat_33)
        mask_ViT_output = torch.sigmoid(mask_ViT_output)

        # import torch.nn.functional as F
        # new_shape = (224, 224)
        # # Resize the array using bilinear interpolation
        # resized_array = F.interpolate(feat_33[3][14].unsqueeze(0), size=new_shape, mode='bilinear',
        #                               align_corners=False).squeeze()
        # summarized_map = np.mean(resized_array.cpu().detach().numpy(), axis=0)
        # # Step 2: Normalize the Summary
        # normalized_map = (summarized_map - np.min(summarized_map)) / (np.max(summarized_map) - np.min(summarized_map))
        # # Step 3: Create a Heatmap
        # plt.imshow(normalized_map)
        # plt.axis('off')
        # plt.colorbar()  # Add color bar for reference
        # plt.savefig('mask_out')  # Save the first image as sa0.png
        # plt.close()

        return ViT_output,feat_3,feat_4,mask_ViT_output

model = VIT_VIT()

def main(pretrained,model_name, N_EPOCHS=100 ,LR = 0.0001, **kwargs ):
    # model = ViT(num_classes=5, pretrained=True)
    # model = Deit(num_classes=5, pretrained=True)
    # model = ResNet50(num_classes=5 , heads=4)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    print(model_name)

    if pretrained:
        print('Continue train:'+model_name)
        print('pred_path:', '/home/linyuxin/pycharm_project/multilabel/ViTmodel_singleeye/weights/'+model_name)
        checkpoint = torch.load('/home/linyuxin/pycharm_project/multilabel/ViTmodel_singleeye/weights/'+model_name)
        state_dict = model.state_dict()
        checkpoint_model = checkpoint
        load_dict = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
        state_dict.update(load_dict)
        model.load_state_dict(state_dict)
        print("model load state dick!")
    else:
        print('Train a new model')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    # scheduler = CosineAnnealingLR(optimizer,T_max=5)
    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=2).to(device)
    criterion = nn.BCELoss().to(device)

    valid_loss = []
    model.to(device)
    best_score = 0
    iter_num=0
    for epoch in range(N_EPOCHS):
        train_epoch_loss = 0.0
        train_semi_loss=0.0
        train_vit_loss=0.0
        train_semi_loss_region=0.0
        if epoch%1==0:
            start_time = datetime.now()
            for start in range(0, len(val_totaldata), BATCH_SIZE):
                end = min(start + BATCH_SIZE,  len(val_totaldata))
                batch_filenames = val_totaldata[start:end]
                local_bz=end-start
                # Calculate class frequencies within the batch
                # set labels
                ori_labels = torch.zeros((local_bz, 6))
                bat_img=[]
                for i, filename in enumerate(batch_filenames):
                    image = Image.open(VAL_PATH + '/Images/' + batch_filenames[i]).convert('RGB')
                    bat_img.append(transform(image))
                    if filename in val_normalones:
                        ori_labels[i, 0] = 1
                    else:
                        if filename in val_age:
                            ori_labels[i, 1] = 1
                        if filename in val_glaucoma:
                            ori_labels[i, 2] = 1
                        if filename in val_diabones:
                            ori_labels[i, 3] = 1
                        if filename in val_cataractones:
                            ori_labels[i, 4] = 1
                        # if filename in val_Hypertension:
                        #     ori_labels[i, 5] = 1
                        if filename in val_Myopia:
                            ori_labels[i, 5] = 1
                ViT_output,feat_1, feat_2,mask_ViT_output= model(torch.stack(bat_img).to(device))
                if start == 0:
                    pred_labels =  ViT_output.cpu().detach()
                    val_labels = ori_labels
                else:
                    pred_labels = np.concatenate(
                        (pred_labels, ViT_output.cpu().detach()),0)
                    val_labels = np.concatenate((val_labels, ori_labels), 0)
            # ... 执行一些操作 ...
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"操作耗时: {duration}")
            scores = evaluate_multilabel_classification(pred_labels, val_labels, threshold=0.5)
            acc_mean = scores['Accuracy Rate per class'].mean()
            f1 = scores["F1-score per class"].mean()
            auc = scores["Average AUC"]
            KS = scores["Kappa score"]
            score = acc_mean + f1 + auc + KS
            print("acc_mean=", acc_mean, 'f1=', f1, 'auc=', auc, 'KS=', KS, 'score=', score)

            if score > best_score:
                patience = 0
                best_model = copy.deepcopy(model.state_dict())
                model.to(device)
                best_score = score
                torch.save(best_model, os.path.join(SAVE_PT_DIR, '{}-{:.4f}.pth'.format(model_name, best_score)))
                print("best model saved, best final score:", best_score)
            else:
                patience = patience + 1
            print('patience=', patience, "best final score:", best_score)







        #data extraction
        np.random.shuffle(train_totaldata)
        total_samples = len(train_totaldata)

        model.train()
        for start in range(0,total_samples,BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_samples)
            batch_filenames = train_totaldata[start:end]
            local_bz=end-start
            # Calculate class frequencies within the batch
            # set labels
            ori_labels = torch.zeros((local_bz, 6))
            class_counts = {'normal': 0, 'age': 0, 'glaucoma': 0, 'diabetes': 0, 'cataract': 0,'Hypertension':0,'Myopia':0}
            for i, filename in enumerate(batch_filenames) :
                if filename in train_normalones:
                    class_counts['normal'] += 1
                    ori_labels[i, 0] = 1
                else:
                    if filename in train_age:
                        class_counts['age'] += 1
                        ori_labels[i, 1] = 1
                    if filename in train_glaucoma:
                        class_counts['glaucoma'] += 1
                        ori_labels[i, 2] = 1
                    if filename in train_diabones:
                        class_counts['diabetes'] += 1
                        ori_labels[i, 3] = 1
                    if filename in train_cataractones:
                        class_counts['cataract'] += 1
                        ori_labels[i, 4] = 1

                    # if filename in train_Hypertension:
                    #     class_counts['Hypertension'] += 1
                    #     ori_labels[i, 5] = 1
                    if filename in train_Myopia:
                        class_counts['Myopia'] += 1
                        ori_labels[i, 5] = 1

            total_classes = sum(class_counts.values())
            augmentation_probs = {cls: 1-(count / total_classes) for cls, count in class_counts.items()}

            condition_outcomes = get_outcomes(augmentation_probs)


            tr_data_cpu=[]
            label_cpu=[]
            for i in range(local_bz):
                image = Image.open(TRAIN_PATH + '/Images/' + batch_filenames[i]).convert('RGB')
                img = transform(image)
                tr_data_cpu.append(img)
                label_cpu.append(ori_labels[i])
                flag=0
                if batch_filenames[i] in train_normalones:
                    flag=flag+condition_outcomes['normal']
                else:
                    if batch_filenames[i] in train_age:
                        flag = flag+condition_outcomes['age']
                    if batch_filenames[i] in train_glaucoma:
                        flag = flag+condition_outcomes['glaucoma']
                    if batch_filenames[i] in train_diabones:
                        flag = flag+condition_outcomes['diabetes']
                    if batch_filenames[i] in train_cataractones:
                        flag = flag+condition_outcomes['cataract']
                    if batch_filenames[i] in train_Hypertension:
                        flag = flag+condition_outcomes['Hypertension']
                    if batch_filenames[i] in train_Myopia:
                        flag = flag+condition_outcomes['Myopia']
                if flag>=0.9:
                    img_aug = transform(transform_images(image))
                    tr_data_cpu.append(img_aug)
                    label_cpu.append(ori_labels[i])
            tr_data=torch.stack(tr_data_cpu).to(device)
            tr_label=torch.stack(label_cpu).to(device)

            # # mixup
            # alpha =0.2
            # lam = np.random.beta(alpha ,alpha)
            # index = torch.randperm(B).to(device)
            # imgs_mix = lam *img + ( 1 -lam ) *img[index]
            #
            # label_a ,label_b = label ,label[index]
            # imgs_mix = imgs_mix.view(-1, C, H, W)
            optimizer.zero_grad()
            ViT_output,feat_1, feat_2,mask_ViT_output= model(tr_data)
            loss_vit = criterion(ViT_output, tr_label)+criterion(mask_ViT_output, tr_label)
            loss_semi=torch.nn.MSELoss()(torch.sigmoid(feat_1), torch.sigmoid(feat_2))
            loss=loss_vit+loss_semi
            #loss = lam *criterion(output, label_a) + ( 1 -lam ) *criterion(output, label_b)



            loss.backward()
            train_epoch_loss += loss.item()
            train_semi_loss += loss_semi.item()
            train_vit_loss += loss_vit.item()
            scheduler.step(epoch + i / train_totaldata.__len__())
            optimizer.step()
            update_ema_variables(model.feature_extractor, model.teacher_feature_extractor, 0.99, iter_num)
            iter_num += 1



        train_loss_mean = train_epoch_loss / train_totaldata.__len__()
        train_semi_loss_mean = train_semi_loss / train_totaldata.__len__()
        train_vit_loss_mean = train_vit_loss / train_totaldata.__len__()
        print(epoch,':\tloss=',train_loss_mean,train_semi_loss_mean,train_vit_loss_mean)


if __name__ == '__main__':
    seed_everything(1001)
    # general global variables
    DATA_PATH = "/home/linyuxin/pycharm_project/multilabel/ViTmodel_singleeye/dataprocess/"
    TRAIN_PATH = DATA_PATH+'TrainingSet'
    TEST_PATH = DATA_PATH+'OnsiteTestSet'
    VAL_PATH = DATA_PATH + 'OffsiteTestSet'
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = './weights'
    NUM_VIEW = 1
    IMAGE_SIZE = (224,224)
    LR = 0.0001
    N_EPOCHS = 300
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 16
    train_label_path = TRAIN_PATH+'/Annotation/training annotation (Chinese).xlsx'
    val_label_path = VAL_PATH + '/Annotation/off-site test annotation (Chinese).xlsx'
    test_label_path = TEST_PATH+'/Annotation/on-site test annotation (Chinese).xlsx'

    train_data=pd.read_excel(train_label_path)
    train_normalones, train_cataractones, train_diabones, train_glaucoma,train_age,train_Hypertension,train_Myopia = dl.process_dataset(train_data)
    train_totaldata=np.concatenate((train_normalones, train_cataractones, train_diabones, train_glaucoma,train_age,train_Myopia), axis=0)

    val_data = pd.read_excel(val_label_path)
    val_normalones, val_cataractones, val_diabones, val_glaucoma, val_age,val_Hypertension,val_Myopia = dl.process_dataset(val_data)
    val_totaldata = np.concatenate((val_normalones, val_cataractones, val_diabones, val_glaucoma, val_age,val_Myopia),
                                     axis=0)

    test_data = pd.read_excel(test_label_path)
    test_normalones, test_cataractones, test_diabones, test_glaucoma, test_age,test_Hypertension,test_Myopia = dl.process_dataset(test_data)
    test_totaldata = np.concatenate((test_normalones, test_cataractones, test_diabones, test_glaucoma, test_age,test_Myopia),
                                     axis=0)

    pretr=0

    if pretr==0:
        mod_name='new_model'
    else:
        mod_name='new_model-3.4208.pth'
    main(pretrained=pretr,model_name=mod_name, N_EPOCHS = N_EPOCHS,LR = LR)

xx=1

# 11 :	loss= 0.21239286009258324 0.19087389029599972 0.02151897003887884
# 操作耗时: 0:00:36.611961
# total_f1= 0.9570815450643777 total_acc= 0.9284116331096197 ks= 0.7414975575639615
# {'Accuracy Rate per class': array([0.80536913, 0.97718121, 0.95704698, 0.84966443, 0.98791946,
#        0.99328859]), 'Precision per class': array([0.74857143, 0.84615385, 0.71428571, 0.88324873, 0.89795918,
#        0.92857143]), 'Recall per class': array([0.82131661, 0.75      , 0.6       , 0.66159696, 0.91666667,
#        0.95121951]), 'F1-score per class': array([0.78325859, 0.79518072, 0.65217391, 0.75652174, 0.90721649,
#        0.93975904]), 'Kappa score': 0.7414975575639615, 'AUC per class': array([0.88006093, 0.96838931, 0.93084892, 0.89464841, 0.99841583,
#        0.99889135]), 'Average AUC': 0.9452091257118019}
# acc_mean= 0.9284116331096196 f1= 0.8056850834955344 auc= 0.9452091257118019 KS= 0.7414975575639615 score= 3.4208033998809175
# best model saved, best final score: 3.4208033998809175
# patience= 0 best final score: 3.4208033998809175
