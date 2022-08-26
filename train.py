import os

import argparse

import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import copy
import random
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Subset
from torchvision import transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter

from utils import Logging
from utils.file_utils import read_yaml, check_dir
from tqdm import tqdm


def train(dataset_info, model_name='efficientnet-b0', save_model_path="./model", batch_size=32, random_seed=555, num_epochs=300):
    writer = SummaryWriter(log_dir=save_model_path)
    image_size = 224

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = EfficientNet.from_pretrained(model_name, num_classes=dataset_info["nc"])
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    train_dataset = torchvision.datasets.ImageFolder(dataset_info["train"], transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(dataset_info["val"], transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    dataloaders, nb_batch = {}, {}
    dataloaders['train'] = train_loader
    dataloaders['valid'] = valid_loader
    nb_batch['train'], nb_batch['valid'] = len(dataloaders['train']), len(dataloaders['valid'])
    print(Logging.i('batch_size : %d,  ' % (batch_size)))
    print(Logging.s('\ttrain dataset batch size(# of image) : %d(%d)' % (nb_batch['train'], nb_batch['train'] * batch_size)))
    print(Logging.s('\tvalid dataset batch size(# of image) : %d(%d)' % (nb_batch['valid'], nb_batch['valid'] * batch_size)))
    device = torch.device("cuda:0")  # set gpu
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(),
                             lr=0.05,
                             momentum=0.9,
                             weight_decay=1e-4)

    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_idx = 0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    for epoch in range(num_epochs):
        print(Logging.i('Epoch {}/{}'.format(epoch, num_epochs - 1)))
        print(Logging.s('-' * 20))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_ft.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                writer.add_scalar("Loss/train", epoch_loss, epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                writer.add_scalar("Loss/valid", epoch_loss, epoch_acc)
            print(Logging.s('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc)))

            if phase == 'valid' and epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(save_model_path, f'epoch-{epoch}.pt'))
                print(Logging.s('Epoch %d model saved' % (epoch)))

            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_model_path, 'best.pt'))
                print(Logging.s('Best model saved - %d / %.1f' % (best_idx, best_acc)))

            torch.save(model.state_dict(), os.path.join(save_model_path, 'last.pt'))
        print(Logging.s('-' * 20))

    time_elapsed = time.time() - start_time
    print(Logging.i('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)))
    print(Logging.i('Best valid Acc: %d - %.1f' % (best_idx, best_acc)))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_model_path, 'final.pt'))
    print(Logging.i('model saved'))

    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-info', type=str, default='data/bookcover-17.yaml', help='dataset_info file path')
    parser.add_argument('--model-name', type=str, default='efficientnet-b0', help='model name')
    parser.add_argument('--dataset', type=str, default='/dataset/bookcover', help='dataset_path')
    parser.add_argument('--save-model-path', type=str, default='weights/efficientnet-b0-bookcover', help='model save directory')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='max number of epoch')

    opt = parser.parse_args()
    dataset_info = read_yaml(opt.dataset_info)
    model_name = opt.model_name
    dataset = opt.dataset
    save_model_path = opt.save_model_path
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs

    check_dir(save_model_path, with_create=True)

    print(Logging.i("Argument Info:"))
    print(Logging.s(f"\tmodel name: {model_name}"))
    print(Logging.s(f"\tdataset info:"))
    print(Logging.s(f"\t\tnumber of class: {dataset_info['nc']}"))
    print(Logging.s(f"\t\ttrain dir: {dataset_info['train']}"))
    print(Logging.s(f"\t\tval dir  : {dataset_info['val']}"))
    print(Logging.s(f"\t\ttest dir : {dataset_info['test']}"))
    print(Logging.s(f"\t\tclass name : {dataset_info['names']}"))
    print(Logging.s(f"\tdataset path: {dataset}"))
    print(Logging.s(f"\tsave model path: {save_model_path}"))
    print(Logging.s(f"\tbatch size: {batch_size}"))
    print(Logging.s(f"\tnum epochs: {num_epochs}"))

    print(Logging.i("Training Start"))
    train(model_name=model_name,
          dataset_info=dataset_info,
          save_model_path=save_model_path,
          batch_size=batch_size,
          num_epochs=num_epochs)