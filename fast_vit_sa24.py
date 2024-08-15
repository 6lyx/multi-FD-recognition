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
device = torch.device('cuda:2')
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



model = timm.create_model('fastvit_sa24.apple_dist_in1k', pretrained=True)
data_config = timm.data.resolve_model_data_config(model)
transforms_m = timm.data.create_transform(**data_config, is_training=False)
model.head.fc=nn.Sequential(
            nn.Linear(1024, 6),
            nn.Sigmoid()
        )



def main(pretrained,model_name, N_EPOCHS=100 ,LR = 0.0001, **kwargs ):
    # model = ViT(num_classes=5, pretrained=True)
    # model = Deit(num_classes=5, pretrained=True)
    # model = ResNet50(num_classes=5 , heads=4)
    transform = transforms_m
    print(model_name)

    if pretrained:
        print('Continue train:'+model_name)
        print('pred_path:', 'weights/'+model_name)
        checkpoint = torch.load('weights/'+model_name)
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
                ViT_output= model(torch.stack(bat_img).to(device))
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
            ViT_output= model(tr_data)
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
    DATA_PATH = "/home/linyuxin/pycharm_project/multilabel/ViTmodel_singleeye/dataprocess/"
    TRAIN_PATH = DATA_PATH+'TrainingSet'
    TEST_PATH = DATA_PATH+'OnsiteTestSet'
    VAL_PATH = DATA_PATH + 'OffsiteTestSet'
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = './weights'
    NUM_VIEW = 1
    IMAGE_SIZE = (224,224)
    LR = 0.0001
    N_EPOCHS = 50
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
        mod_name='fastvit_sa24'
    else:
        mod_name='new_model-3.4146.pth'
    main(pretrained=pretr,model_name=mod_name, N_EPOCHS = N_EPOCHS,LR = LR)

xx=1

# 11 :	loss= 0.006948100174717717 0.0 0.006948100174717717
# 操作耗时: 0:01:53.084000
# total_f1= 0.9416509814466253 total_acc= 0.9029082774049217 ks= 0.6527272824913674
# {'Accuracy Rate per class': array([0.74228188, 0.96644295, 0.9557047 , 0.77181208, 0.98791946,
#        0.99328859]), 'Precision per class': array([0.68194842, 0.75675676, 0.77419355, 0.70852018, 0.88235294,
#        0.89130435]), 'Recall per class': array([0.7460815 , 0.63636364, 0.48      , 0.60076046, 0.9375    ,
#        1.        ]), 'F1-score per class': array([0.71257485, 0.69135802, 0.59259259, 0.65020576, 0.90909091,
#        0.94252874]), 'Kappa score': 0.6527272824913674, 'AUC per class': array([0.82123567, 0.94530541, 0.90166906, 0.83604437, 0.99754902,
#        0.99906458]), 'Average AUC': 0.916811350852424}
# acc_mean= 0.9029082774049217 f1= 0.7497251456038861 auc= 0.916811350852424 KS= 0.6527272824913674 score= 3.222172056352599
# best model saved, best final score: 3.222172056352599
# patience= 0 best final score: 3.222172056352599