from PIL import Image
import sys
import json
import time
from random import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import matplotlib.image as mpimg
from torchvision import transforms, datasets
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torchvision
import cv2 as cv


def get_dataloader(args):  # 读取数据集

    data_transform = {  # 数据集的预处理
        'train': transforms.Compose([transforms.RandomResizedCrop(244),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(args.ImgPath, "train"),
                                         transform=data_transform['train'])  # 指定训练集地址、预处理等
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=4)  # 加载训练集数据

    val_dataset = datasets.ImageFolder(root=os.path.join(args.ImgPath, "valid"),
                                       transform=data_transform['val'])  # 指定测试集地址、预处理等
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=4)  # 加载训练集数据

    emotion_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in emotion_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=args.the_classes)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    data_siuation = {'train_data': train_dataset, 'train_loader': train_loader,
                     'val_data': val_dataset, 'val_loader': val_loader}
    return data_siuation


def get_net(args):  # 定义使用哪个网络模型、预训练、微调等
    seed = 42
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    np.random.seed = seed
    torch.cuda.manual_seed_all(seed)  # 设置随机数种子，使实验能重复
    if args.net_choice == 'resnet101':
        net = models.resnet101(pretrained=True)
        # net = models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)  # 使用哪个网络
        model_weight_path = args.model_path
        net.load_state_dict(torch.load(model_weight_path), strict=False)  # 使用哪个预训练模型

        num_classes = args.the_classes
        in_channel = net.fc.in_features  # Resnet的微调，最后一层输出改为num_classes个
        net.fc = torch.nn.Linear(in_channel, num_classes)

    if args.net_choice == 'vgg19':
        net = models.vgg19(pretrained=True)
        # net = models.vgg19(pweights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        model_weight_path = args.model_path
        net.load_state_dict(torch.load(model_weight_path), strict=False)  # 使用哪个预训练模型

        num_classes = args.the_classes
        net.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes))

    return net


def train_and_eval(args, net, data_siuation):
    torch.cuda.empty_cache()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 调用GPU
    print("using {} device.".format(device))
    net.to(device)
    epochs = args.epoch

    # 优化器和学习率
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=args.lr,
                                    weight_decay=0)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=args.lr,
                                     weight_decay=0)
    elif args.optimizer_type == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=args.lr,
                                      weight_decay=0)
    elif args.optimizer_type == 'ranger':
        optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=args.lr,
                           weight_decay=0)

    if args.scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50, 70, 90, 110, 130, 150, 180],
                                                         gamma=0.1, last_epoch=-1)
    # 损失函数
    if args.loss == 'CrossEntropy':
        loss_funtion = nn.CrossEntropyLoss()

    save_path = args.savepath  # 保存训练中较好的pth
    train_dataset = data_siuation['train_data']
    train_loader = data_siuation['train_loader']
    val_dataset = data_siuation['val_data']
    val_loader = data_siuation['val_loader']

    len_train = len(train_dataset)  # 训练集图像数量
    len_val = len(val_dataset)  # 测试集图像数量
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    train_losses = []
    val_losses = []
    train_acces = []
    val_acces = []
    best_acc = 0.0
    fit_time = time.time()
    print(f"using {len_train} images for training, {len_val} images for validation.")
    for e in range(epochs):
        since = time.time()
        training_loss = 0
        training_acc = 0
        with tqdm(total=len(train_loader)) as pbar:
            for train_image, train_label in train_loader:
                net.train()
                optimizer.zero_grad()
                train_image = train_image.to(device)
                train_label = train_label.to(device)
                # forward

                output = net(train_image)  # 网络输出结果
                Loss = loss_funtion(output, train_label)  # 将输出结果与标签做对比计算误差
                pred_train = torch.max(output, dim=1)[1]
                # backward
                Loss.backward()
                optimizer.step()  # update weight
                #scheduler.step()  # 更新学习率

                training_loss += Loss.item()
                training_acc += torch.eq(pred_train,train_label).sum().item()
                # pbar.desc = "train epoch[{}/{}] train_loss:{:.3f}".format(e + 1,
                #                                                                epochs,
                #                                                                Loss)
                pbar.update(1)

        net.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for val_image, val_label in val_loader:
                    val_image = val_image.to(device)
                    val_label = val_label.to(device)
                    output = net(val_image)

                    Loss = loss_funtion(output,val_label)
                    pred_val = torch.max(output, dim=1)[1]
                    val_loss += Loss.item()
                    val_acc += torch.eq(pred_val,val_label).sum().item()
                    pb.update(1)

        train_acces.append(training_acc / len_train)  # 计算总的训练集正确率
        train_losses.append(training_loss / train_steps)  # 计算总的训练误差

        val_acces.append(val_acc / len_val)  # 计算总的训练集正确率
        val_losses.append(val_loss / val_steps)  # 计算总的训练误差

        print("total epoch:{}/{}    ".format(e + 1, epochs),
              "Train acc :{:3f}    ".format(training_acc / len_train),
              "Train loss: {:3f}    ".format(training_loss / train_steps),
              "Val acc:{:3f}    ".format(val_acc / len_val),
              "Val loss:{:3f}    ".format(val_loss / val_steps),
              "Time: {:.2f}s".format((time.time() - since))
              )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print(" train_acc_avg is {}".format(np.mean(train_acces, 0)),
          " train_loss_avg is {}".format(np.mean(train_losses, 0)),
          " val_acc_avg is {}".format(np.mean(val_acces, 0)),
          " val_loss_avg is {}".format(np.mean(val_losses, 0)),
          ' Total time: {:.2f} hours'.format((time.time() - fit_time) / 3600)
          )


# 使用训练好的网络对单张图像分类的效果
def predict_one(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 图像预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_path = args.img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    net = models.resnet101(num_classes=args.the_classes).to(device)

    # load model weights
    weights_path = args.savepath
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path))

    net.eval()
    with torch.no_grad():
        #output = net(val_images.to(device))  # 将图片输入网络 获取结果
        output = torch.squeeze(net(img.to(device)))
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).cpu().numpy()

    print_result = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                    predict[predict_class].cpu().numpy())

    parent_folder_name = os.path.basename(os.path.dirname(img_path ))
    plt.title(f'real class:  {parent_folder_name} \n  predict {print_result}')
    for i in range(len(predict)):
        print("classes : {:10}  probability: {:3}".format(class_indict[str(i)],
                                                          predict[i].cpu().numpy()))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-BATCH_SIZE', type=int, default=32, help='batch size for dataloader')  # batch_size
    parser.add_argument('-lr', type=int, default=0.0001, help='learning rate')  # learning rate
    parser.add_argument('-model_path', type=str, default='YOUR FILE ADDRESS',
                        help='path of model')  # model_path
    parser.add_argument('-the_classes', type=int, default=8, help='num of classes for model')  # classes of image
    parser.add_argument('-net_choice', type=str, default='resnet101', help='net type')  # choose which model to train
    parser.add_argument('-optimizer_type', type=str, default='adam', help='the_optimizer')  # choose an optimizer
    parser.add_argument('-scheduler_type', type=str, default='MultiStepLR',
                        help='changing the learning rate')  # change lr
    parser.add_argument('-epoch', type=int, default=200, help='the times of train epoch')  # epoch
    parser.add_argument('-savepath', type=str, default='YOUR FILE ADDRESS',
                        help='the way to save model ')  # save a new model
    #parser.add_argument('-img_path', type=str, default='YOUR FILE ADDRESS',
                        help='single predict image path ')  # to predict single image
    parser.add_argument('-loss', type=str, default="CrossEntropy", help='loss function')  # loss function

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "YOUR FILE ADDRESS")  # emotion6 data set path
    parser.add_argument('-ImgPath', type=str, default="YOUR FILE ADDRESS", help='the path of image dataset')

    args = parser.parse_args()
    net = get_net(args)  # 获取网络
    data_siuation = get_dataloader(args)  # 获取数据

    train_and_eval(args, net, data_siuation)  # 送入训练
    #predict_one(args)
    #predict_one(args)
    # imgpath = 'E:\\DL_dataset\\artphoto\\valid\\anger\\anger_0002.jpg'
    # # img = cv.imread(imgpath)
    # # cv.imshow('555',img)
    # # cv.waitKey(50)  # 延时显示
    # img = mpimg.imread(imgpath)
    #
    # plt.imshow(img)
    # plt.show()