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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, roc_auc_score

import timm

import torch


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


from transformers import AutoFeatureExtractor, CvtForImageClassification
from PIL import Image

feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/cvt-21')
model = CvtForImageClassification.from_pretrained('microsoft/cvt-21')
model.classifier=nn.Sequential(
            nn.Linear(384, 6),
            nn.Sigmoid()
        )



def main(pretrained,model_name, N_EPOCHS=100 ,LR = 0.0001, **kwargs ):
    # model = ViT(num_classes=5, pretrained=True)
    # model = Deit(num_classes=5, pretrained=True)
    # model = ResNet50(num_classes=5 , heads=4)
    transform = feature_extractor
    print(model_name)

    if pretrained:
        print('Continue train:'+model_name)
        print('pred_path:', 'weights\\'+model_name)
        checkpoint = torch.load('weights\\'+model_name)
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
            # for start in range(0, len(val_totaldata), BATCH_SIZE):
            #     end = min(start + BATCH_SIZE,  len(val_totaldata))
            #     batch_filenames = val_totaldata[start:end]
            #     local_bz=end-start
            #     # Calculate class frequencies within the batch
            #     # set labels
            #     ori_labels = torch.zeros((local_bz, 6))
            #     bat_img=[]
            #     for i, filename in enumerate(batch_filenames):
            #         image = Image.open(VAL_PATH + '/Images/' + batch_filenames[i]).convert('RGB')
            #         bat_img.append(transform(image))
            #         if filename in val_normalones:
            #             ori_labels[i, 0] = 1
            #         else:
            #             if filename in val_age:
            #                 ori_labels[i, 1] = 1
            #             if filename in val_glaucoma:
            #                 ori_labels[i, 2] = 1
            #             if filename in val_diabones:
            #                 ori_labels[i, 3] = 1
            #             if filename in val_cataractones:
            #                 ori_labels[i, 4] = 1
            #             # if filename in val_Hypertension:
            #             #     ori_labels[i, 5] = 1
            #             if filename in val_Myopia:
            #                 ori_labels[i, 5] = 1
            #     ViT_output,feat_1, feat_2,mask_ViT_output= model(torch.stack(bat_img).to(device))
            #     if start == 0:
            #         pred_labels =  ViT_output.cpu().detach()
            #         val_labels = ori_labels
            #     else:
            #         pred_labels = np.concatenate(
            #             (pred_labels, ViT_output.cpu().detach()),0)
            #         val_labels = np.concatenate((val_labels, ori_labels), 0)
            # # ... 执行一些操作 ...
            # end_time = datetime.now()
            # duration = end_time - start_time
            # print(f"操作耗时: {duration}")
            # scores = evaluate_multilabel_classification(pred_labels, val_labels, threshold=0.5)
            # acc_mean = scores['Accuracy Rate per class'].mean()
            # f1 = scores["F1-score per class"].mean()
            # auc = scores["Average AUC"]
            # KS = scores["Kappa score"]
            # score = acc_mean + f1 + auc + KS
            # print("acc_mean=", acc_mean, 'f1=', f1, 'auc=', auc, 'KS=', KS, 'score=', score)
            for start in range(0, len(test_totaldata), BATCH_SIZE):
                end = min(start + BATCH_SIZE,  len(test_totaldata))
                batch_filenames = test_totaldata[start:end]
                local_bz=end-start
                # Calculate class frequencies within the batch
                # set labels
                ori_labels = torch.zeros((local_bz, 6))
                bat_img=[]
                for i, filename in enumerate(batch_filenames):
                    image = Image.open(TEST_PATH + '/Images/' + batch_filenames[i]).convert('RGB')
                    bat_img.append(transform(image))
                    if filename in test_normalones:
                        ori_labels[i, 0] = 1
                    else:
                        if filename in test_age:
                            ori_labels[i, 1] = 1
                        if filename in test_glaucoma:
                            ori_labels[i, 2] = 1
                        if filename in test_diabones:
                            ori_labels[i, 3] = 1
                        if filename in test_cataractones:
                            ori_labels[i, 4] = 1
                        # if filename in test_Hypertension:
                        #     ori_labels[i, 5] = 1
                        if filename in test_Myopia:
                            ori_labels[i, 5] = 1
                ViT_output,feat_1, feat_2,mask_ViT_output= model(torch.stack(bat_img).to(device))
                if start == 0:
                    pred_labels =  ViT_output.cpu().detach()
                    test_labels = ori_labels
                else:
                    pred_labels = np.concatenate(
                        (pred_labels, ViT_output.cpu().detach()),0)
                    test_labels = np.concatenate((test_labels, ori_labels), 0)
            # ... 执行一些操作 ...
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"操作耗时: {duration}")
            scores = evaluate_multilabel_classification(pred_labels, test_labels, threshold=0.5)
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
                image = Image.open(TRAIN_PATH + '\Images\\' + batch_filenames[i]).convert('RGB')
                img = image
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
                    img_aug = transform_images(image)
                    tr_data_cpu.append(img_aug)
                    label_cpu.append(ori_labels[i])
            tr_data=torch.Tensor(np.stack(transform(tr_data_cpu).data['pixel_values'])).to(device)

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
            ViT_output= model(tr_data).logits
            loss_vit = criterion(ViT_output, tr_label)

            loss=loss_vit
            #loss = lam *criterion(output, label_a) + ( 1 -lam ) *criterion(output, label_b)



            loss.backward()
            train_epoch_loss += loss.item()
            train_vit_loss += loss_vit.item()
            scheduler.step(epoch + i / train_totaldata.__len__())
            optimizer.step()
            iter_num += 1



        train_loss_mean = train_epoch_loss / train_totaldata.__len__()
        train_semi_loss_mean = train_semi_loss / train_totaldata.__len__()
        train_vit_loss_mean = train_vit_loss / train_totaldata.__len__()
        print(epoch,':\tloss=',train_loss_mean,train_semi_loss_mean,train_vit_loss_mean)


if __name__ == '__main__':
    seed_everything(1001)
    # general global variables
    DATA_PATH = "F:\python_project\multi_eye_disease_classification\mycode\\fuwuqi\ViTmodel_singleeye\dataprocess\\"
    TRAIN_PATH = DATA_PATH+'TrainingSet'
    TEST_PATH = DATA_PATH+'OnsiteTestSet'
    VAL_PATH = DATA_PATH + 'OffsiteTestSet'
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = '.\weights'
    NUM_VIEW = 1
    IMAGE_SIZE = (224,224)
    LR = 0.0001
    N_EPOCHS = 100
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 16
    train_label_path = TRAIN_PATH+'\Annotation\\training annotation (Chinese).xlsx'
    val_label_path = VAL_PATH + '\Annotation\off-site test annotation (Chinese).xlsx'
    test_label_path = TEST_PATH+'\Annotation\on-site test annotation (Chinese).xlsx'

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

    pretr=1

    if pretr==0:
        mod_name='cvt_21'
    else:
        mod_name='cvt_13-3.0616.pth'
    main(pretrained=pretr,model_name=mod_name, N_EPOCHS = N_EPOCHS,LR = LR)

xx=1

# total_f1= 0.9479348849724203 total_acc= 0.9134228187919463 ks= 0.6911478781581244
# {'Accuracy Rate per class': array([0.76241611, 0.97852349, 0.95167785, 0.80402685, 0.99060403,
#        0.99328859]), 'Precision per class': array([0.69189189, 0.91176471, 0.79166667, 0.76470588, 0.91836735,
#        0.90909091]), 'Recall per class': array([0.80250784, 0.70454545, 0.38      , 0.64258555, 0.9375    ,
#        0.97560976]), 'F1-score per class': array([0.74310595, 0.79487179, 0.51351351, 0.69834711, 0.92783505,
#        0.94117647]), 'Kappa score': 0.6911478781581244, 'AUC per class': array([0.83936009, 0.92556089, 0.85510791, 0.85182147, 0.99835605,
#        0.99913387]), 'Average AUC': 0.9115567125725668}
# acc_mean= 0.9134228187919463 f1= 0.7698083147685121 auc= 0.9115567125725668 KS= 0.6911478781581244 score= 3.2859357242911496
# best model saved, best final score: 3.2859357242911496
# patience= 0 best final score: 3.2859357242911496