from __future__ import print_function, division
import torch
import torch.nn as nn #파이토치 nn모듈 임포트
import torch.optim as optim # PyTorch optim패키지 임포트
from torch.optim import lr_scheduler # 학습률 스케쥴러 임포트
import numpy as np # 넘파이 임포트
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torchsummary import summary

import seaborn as sns

# plt.ion()   # 대화형 모드

data_transforms = {
    'train': transforms.Compose([  # 이미지 변형시키기
        transforms.Resize((224, 224)),
        transforms.RandomCrop(150),  # 랜덤으로 자름
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색 변조
        transforms.RandomHorizontalFlip(p=1),  # 수평으로 이미지 뒤집기
        transforms.ToTensor(),  # 이미지 데이터를 tensor로 바꿔준다.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(200), # 이미지 중앙을 resize × resize로 자른다
        transforms.ToTensor(),  # 이미지 데이터를 tensor로 바꿔준다.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화

    ]),

    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(200), # 이미지 중앙을 resize × resize로 자른다
        transforms.ToTensor(),  # 이미지 데이터를 tensor로 바꿔준다.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화

    ]),

}

data_dir = 'C:/Users/user/PycharmProjects/machineLearningProject/Data'  # train, val, test 경로설정 # 절대경로로 작업하여 다른 곳에서 작업시 경로 수정 필요

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),  # 설정한 경로에서 이미지를 가져와 리사이즈해서 데이터저장
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,  # 배치사이즈 16
                                              shuffle=True, num_workers=2)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu가 사용불가일 때 cpu를 사용.
# 해당 작업을 위해 GPU가 있는 노트북을 구매한 후 cuda 설치를 했지만 작업상의 오류로 cuda가 인식되지 않음.
# 차후 작업은 cpu로 진행함.
# cpu로 진행함에 따라 학습을 간소화하여 결과를 냄.


def imshow(inp, title=None):  # 사용할 이미지의 일부를 보여줌(train)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.close(inp)


def visualize_model(model, num_images=7):  # val 일부 이미지에대한 예측 값을 보여주는 함수
    was_training = model.training
    model.eval()  # 모델을 검증모드로
    images_so_far = 0
    fig = plt.figure()  # figure를 만들고 편집 할 수 있게 만들어주는 함수

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 3, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))  # 가장 높은확률의 이름 출력
                if torch.cuda.is_available(): # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    imshow(inputs.cpu().data[j])  # 예측하려고 입력된 이미지 보여주기
                plt.cla()
                if not torch.cuda.is_available():
                    plt.rcParams.update({'figure.max_open_warning': 0}) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # plt.close(fig)
                    if not torch.cuda.is_available():
                        plt.rcParams.update({'figure.max_open_warning': 0}) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    return
        model.train(mode=was_training)


def train_model(model, criterion, optimizer, num_epochs=50):  # training 함수 정의, num_epochs 조정
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            # if phase == 'train':
            #    scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)  # accuracy best model 을 저장 한다.
    torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/machineLearningProject/MODELS/non_reg1_2.pt')  # 모델을 저장할 자신의 경로 설정 # 절대경로로 작업하여 다른 곳에서 작업시 경로 수정 필요
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


def test_visualize_model(model, num_images=4):  # test 일부 이미지에대한 예측값을 보여주는 함수
    was_training = model.training
    model.eval()  # 모델을 검증모드로
    images_so_far = 0
    fig = plt.figure()  # figure를 만들고 편집 할 수 있게 만들어주는 함수

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 3, images_so_far) # 7종의 데이터를 보여줄 수 있도록 작업.
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))  # 가장 높은확률의 이름 출력
                if torch.cuda.is_available(): # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    imshow(inputs.cpu().data[j])  # 예측하려고 입력된 이미지 보여주기
                plt.cla()
                if not torch.cuda.is_available():
                    plt.rcParams.update({'figure.max_open_warning': 0}) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # plt.close(fig) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    plt.rcParams.update({'figure.max_open_warning': 0}) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    return
        model.train(mode=was_training)


def test_visualize_model(model, num_images=2):  # test Image  예측값을 보여주는 함수
    was_training = model.training
    model.eval()  # 모델을 검증모드로
    images_so_far = 0
    fig = plt.figure()  # figure를 만들고 편집 할 수 있게 만들어주는 함수

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 3, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))  # 가장 높은확률의 이름 출력
                if torch.cuda.is_available(): # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    imshow(inputs.cpu().data[j])  # 예측하려고 입력된 이미지 보여주기
                plt.cla()
                if not torch.cuda.is_available():
                    plt.rcParams.update({'figure.max_open_warning': 0}) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # plt.close(fig) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    if not torch.cuda.is_available():
                        plt.rcParams.update({'figure.max_open_warning': 0}) # cpu로 해당 데이터를 모두 보여줄 경우 에러 발생하여 처리함.
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    plt.ion()  # 대화형 모드

    inputs, classes = next(iter(dataloaders['train']))  # 학습 데이터의 배치를 얻습니다.
    out = torchvision.utils.make_grid(inputs)  # 배치로부터 격자 형태의 이미지를 만듭니다.

    # imshow(out, title=[class_names[x] for x in classes])  # 이미지 보여주기

    inputs, classes = next(iter(dataloaders['val']))  # 학습 데이터의 배치를 얻습니다.
    out = torchvision.utils.make_grid(inputs)  # 배치로부터 격자 형태의 이미지를 만듭니다.

    # imshow(out, title=[class_names[x] for x in classes])  # 이미지 보여주기

    model = models.regnet_x_32gf(pretrained=True)  # models 사용가능 모델들(regnet 이용) : https://pytorch.org/vision/stable/models.html 참고
    print(model)
    model.fc = nn.Linear(in_features=2520, out_features=7)  # 마지막 출력층을 나의 class 수에 맞춰서 바꿔준다.
    print(model)  # 바뀐모델 구조 출력

    model = model.to(device)  # 모델을 gpu로 바꾸는 부분이나 cuda가 인식이 되지 않아 cpu로 작업함.
    criterion = nn.CrossEntropyLoss()  # 손실함수(loss function) 크로스 엔트로피 사용
    # 최적화 기법 설정
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # optimizer sgd, 학습률 0.001

    summary(model, input_size=(3, 224, 224))

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion, optimizer,
                                                                                          num_epochs=5)
    # 모델 시각화 train, val 의 accuracy , loss 시각화
    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.show()
    for x in range(3):
        visualize_model(model)  # val 이미지 모델 예측값 시각화

    for x in range(10):
        test_visualize_model(model)  # test 이미지 모델 예측값 시각화

    nb_classes = 7 # K3, K5, NIRO, SANTAFE, SONATA, TUCSON, GRANDEUR : 7 차종

    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('        K3', '    K5', '   NIRO', ' SANTAFE', 'SONATA', 'TUCSON', 'GRANDEUR')
    print(confusion_matrix.diag() / confusion_matrix.sum(1))

    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, fmt='g',
                ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['K3', 'K5', 'NIRO', 'SANTAFE', 'SONATA', 'TUCSON', 'GRANDEUR']); # 7종류의 차량 라벨
    ax.yaxis.set_ticklabels(['K3', 'K5', 'NIRO', 'SANTAFE', 'SONATA', 'TUCSON', 'GRANDEUR']); # 7종류의 차량 라벨

    model = models.regnet_x_32gf(pretrained=False)
    model.fc = nn.Linear(in_features=2520, out_features=7)

    model.load_state_dict(torch.load("C:/Users/user/PycharmProjects/machineLearningProject/MODELS/non_reg1_2.pt", # 절대경로로 작업하여 다른 곳에서 작업시 경로 수정 필요
                                     map_location=torch.device('cpu')))
    model = model.to(device)

    for x in range(20):  # 2개씩 20번 반복 40장
        test_visualize_model(model)  # test 이미지 모델 예측값 시각화

