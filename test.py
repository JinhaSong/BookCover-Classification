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

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    print(Logging.i("Start evaluation"))
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(Logging.i("Result"))
    print(Logging.s("-" * 20))
    for i, (classname, correct_count) in enumerate(correct_pred.items()):
        str_classname = str(classname).ljust(20)
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(Logging.s(f'{str_classname}: {accuracy:.1f} %'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info', type=str, default='data/bookcover-17.yaml', help='dataset_path')
    parser.add_argument('--model-path', type=str, default='model/best.pt', help='model name')
    parser.add_argument('--batch-size', type=int, default=32, help='model name')

    opt = parser.parse_args()
    dataset_info = read_yaml(opt.dataset_info)
    batch_size = opt.batch_size
    model_path = opt.model_path
    test(model_path, dataset_info, batch_size)