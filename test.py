import os
import argparse
import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Subset, Dataset
from torchvision import transforms

from utils import Logging
from utils.file_utils import read_yaml
from torchvision import datasets
from tqdm import tqdm


def test(model_path, dataset_info, batch_size=32, model_name='efficientnet-b0'):
    image_size = 224
    device = torch.device("cuda:0")
    print(Logging.i(f"Load model({model_name})"))
    model = EfficientNet.from_pretrained(model_name, num_classes=dataset_info['nc'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    print(Logging.i("Load test dataset"))
    transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    testset = datasets.ImageFolder(dataset_info['test'], transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)
    classes = dataset_info["names"]

    top1_pred = {classname: 0 for classname in classes}
    top5_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    print(Logging.i("Start evaluation"))
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.topk(outputs, k=1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    top1_pred[classes[label]] += 1

            _, predictions = torch.topk(outputs, k=5)
            for label, prediction in zip(labels, predictions):
                if label in prediction:
                    top5_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(Logging.i("Result"))
    print(Logging.s("| class                |  top1  |  top5  |"))
    print(Logging.s("-" * 43))
    top1_avg_acc = 0
    top5_avg_acc = 0
    for (top1_classname, top1_correct_count), (top5_classname, top5_correct_count) in zip(top1_pred.items(), top5_pred.items()):
        str_classname = str(top1_classname).ljust(20)
        top1_accuracy = 100 * float(top1_correct_count) / total_pred[top1_classname]
        top5_accuracy = 100 * float(top5_correct_count) / total_pred[top5_classname]
        top1_avg_acc += top1_accuracy
        top5_avg_acc += top5_accuracy

        print(Logging.s("| {} | {}% | {}% |".format(str_classname, str(round(top1_accuracy, 1)).ljust(5), str(round(top5_accuracy, 1)).ljust(5))))
    print(Logging.s("-" * 43))
    print(Logging.s("| {} | {}% | {}% |".format("Average".ljust(20), str(round(top1_avg_acc/len(classes), 1)).ljust(5), str(round(top5_avg_acc/len(classes), 1)).ljust(5))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='efficientnet-b0', help='model name')
    parser.add_argument('--dataset-info', type=str, default='data/bookcover-17.yaml', help='dataset_path')
    parser.add_argument('--model-path', type=str, default='model/best.pt', help='model name')
    parser.add_argument('--batch-size', type=int, default=32, help='model name')

    opt = parser.parse_args()
    model_name = opt.model_name
    dataset_info = read_yaml(opt.dataset_info)
    batch_size = opt.batch_size
    model_path = opt.model_path
    test(model_name, model_path, dataset_info, batch_size)