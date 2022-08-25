import os.path

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

from utils import Logging
from utils.file_utils import read_yaml


def train(dataset_info, data_path="./dataset", save_model_path="./model", batch_size=32, random_seed=555, num_epochs=300):
    image_size = 224

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=dataset_info["nc"])
    bookcover_dataset = torchvision.datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    datasets = {}
    train_idx, tmp_idx = train_test_split(list(range(len(bookcover_dataset))), test_size=0.2, random_state=random_seed)
    datasets['train'] = Subset(bookcover_dataset, train_idx)
    tmp_dataset = Subset(bookcover_dataset, tmp_idx)
    val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
    datasets['valid'] = Subset(tmp_dataset, val_idx)
    datasets['test'] = Subset(tmp_dataset, test_idx)

    dataloaders, nb_batch = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=1)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=1)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=1)
    nb_batch['train'], nb_batch['valid'], nb_batch['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(dataloaders['test'])
    print(Logging.i('batch_size : %d,  train valid test : %d / %d / %d\n' % (batch_size, nb_batch['train'], nb_batch['valid'], nb_batch['test'])))

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

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
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
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print(Logging.i('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc)))

            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_model_path, 'best.pt'))
                print(Logging.i('==> best model saved - %d / %.1f' % (best_idx, best_acc)))

            torch.save(model.state_dict(), os.path.join(save_model_path, 'last.pt'))

    time_elapsed = time.time() - start_time
    print(Logging.i('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)))
    print(Logging.i('Best valid Acc: %d - %.1f' % (best_idx, best_acc)))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_model_path, 'final.pt'))
    print(Logging.i('model saved'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info', type=str, default='data/bookcover-17.yaml', help='dataset_path')
    parser.add_argument('--dataset', type=str, default='/dataset/bookcover', help='dataset_path')
    parser.add_argument('--save-model-path', type=str, default='./model', help='model name')
    parser.add_argument('--batch-size', type=int, default=32, help='model name')
    parser.add_argument('--random-seed', type=int, default=555, help='model name')
    parser.add_argument('--num-epochs', type=int, default=50, help='model name')

    opt = parser.parse_args()
    dataset_info = read_yaml(opt.dataset_info)
    dataset = opt.dataset
    save_model_path = opt.save_model_path
    batch_size = opt.batch_size
    random_seed = opt.random_seed
    num_epochs = opt.num_epochs
    print(Logging.i("Argument Info:"))
    print(Logging.s(f"\tdataset info:"))
    print(Logging.s(f"\t\tnumber of class: {dataset_info['nc']}"))
    print(Logging.s(f"\t\ttrain dir: {dataset_info['train']}"))
    print(Logging.s(f"\t\tval dir  : {dataset_info['val']}"))
    print(Logging.s(f"\t\ttest dir : {dataset_info['test']}"))
    print(Logging.s(f"\t\tclass name : {dataset_info['names']}"))
    print(Logging.s(f"\tdataset path: {dataset}"))
    print(Logging.s(f"\tsave model path: {save_model_path}"))
    print(Logging.s(f"\tbatch_size: {batch_size}"))
    print(Logging.s(f"\trandom_seed: {random_seed}"))
    print(Logging.s(f"\tnum_epochs: {num_epochs}"))

    print(Logging.i("Training Start"))
    train(dataset_info=dataset_info,
          data_path=dataset,
          save_model_path=save_model_path,
          batch_size=batch_size,
          random_seed=random_seed,
          num_epochs=num_epochs)