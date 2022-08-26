import os
import json

import argparse
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from utils import Logging
from utils.file_utils import read_yaml

def detect(dataset_info, model_name='efficientnet-b0', model_path='model/best.pt', image_path="test.jpg", topk=5):
    print(Logging.i(f"Load model({model_name})"))
    model = EfficientNet.from_pretrained(model_name, num_classes=dataset_info['nc'])
    model.load_state_dict(torch.load(model_path))

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    print(Logging.i("Load image"))
    img = transform(Image.open(image_path)).unsqueeze(0)

    labels_map = dataset_info["names"]

    print(Logging.i("Start inference"))
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    print(Logging.i("Inference Result"))
    for idx in torch.topk(outputs, k=topk).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print(Logging.s('\t{idx:>3} - {label:<20}: {p:.2f}%'.format(idx=idx+1, label=labels_map[idx], p=prob*100)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-info', type=str, default='data/bookcover-17.yaml', help='dataset info file path')
    parser.add_argument('--model-name', type=str, default='efficientnet-b0', help='model name')
    parser.add_argument('--model-path', type=str, default='model/best.pt', help='model path')
    parser.add_argument('--image-path', type=str, default='/dataset/bookcover/test/adult/adult_000000.jpg', help='image path')
    parser.add_argument('--topk', type=int, default=5, help='top k')

    opt = parser.parse_args()
    model_name = opt.model_name
    model_path = opt.model_path
    dataset_info = read_yaml(opt.dataset_info)
    image_path = opt.image_path
    topk = opt.topk
    detect(dataset_info=dataset_info, model_name=model_name, model_path=model_path, image_path=image_path, topk=topk)