# BookCover-Classification

## Contents
* [Requirements](#Requirements)
* [Dataset](#Dataset)
* [Train](#Train)
* [Test](#Test)
* [Detect](#Detect)

## Requirements
* Ubuntu 20.04
* CUDA 11.1
* install pip requirements
  * ```pip install -r requirements.txt```

## Dataset
* [Link](docs/Rename.md) 참고

## Train
* [Dataset](#Dataset)과 같이 데이터셋을 구축했다면 아래 명령어를 통해 학습 가능
* 아래 성능은 bookcover-17 데이터셋 기반이며 epoch 50까지의 best accuracy 모델 기준
### 모델 목록
  * 이 프로젝트는 [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) 기반으로 구현되어 있음
  * Accuracy detail : [docs/perform_detail.md](docs/perform_detail.md)
  
|    *Name*         |*# Params*| *Top-1 Acc.* | *Top-5 Acc.* |
|:-----------------:|:--------:|:------------:|:------------:|
| `efficientnet-b0` |   5.3M   |   35.9 %     |   73.8 %     |
| `efficientnet-b1` |   7.8M   |              |              |
| `efficientnet-b2` |   9.2M   |              |              |
| `efficientnet-b3` |    12M   |              |              |
| `efficientnet-b4` |    19M   |              |              |
| `efficientnet-b5` |    30M   |              |              |
| `efficientnet-b6` |    43M   |              |              |
| `efficientnet-b7` |    66M   |              |              |

### 학습 방법
* 아래 명령어를 이용해 학습 가능
    ```shell
    python train.py \
            --model-name=efficientnet-b7 \
            --dataset-info=data/bookcover-17.yaml \
            --dataset=/dataset/bookcover \
            --save-model-path=weights/efficientnet-b0-bookcover \
            --batch-size=8 \
            --num-epochs=50
    ```
  * ```--model-name```: [모델 목록](#모델 목록)에 있는 모델 중 하나로 선택
  * ```--dataset-info```: [Dataset](#Dataset)에서 생성한 ```${DATASET}.yaml``` 파일 
  * ```--dataset```: [Dataset](#Dataset)에서 생성한 데이터셋 경로
  * ```--save-model-path```: 학습한 모델을 저장할 디렉토리 경로(디렉토리 없을 시 생성함)
  * ```--batch-size```: batch size
  * ```--num-epoch```: 최대 학습 epoch 수
## Test
* 학습한 모델로 ```${DATASET}.yaml``` 파일의 testset에 대하여 top1, top5 성능을 추출
  ```shell
  python test.py \
          --model-name=efficientnet-b0 \
          --dataset-info=data/bookcover-17.yaml \
          --model-path=weights/efficientnet-b0-bookcover.pt \
          --batch-size=32
  ```
  * 출력 결과
    ```shell
    [2022-08-26 14:18:47.900422] -     INFO: Load model(efficientnet-b0)
    Loaded pretrained weights for efficientnet-b0
    [2022-08-26 14:18:58.718558] -     INFO: Load test dataset
    [2022-08-26 14:18:58.790107] -     INFO: Start evaluation
    100%|█████████████████████████████████████████| 279/279 [01:00<00:00,  4.63it/s]
    [2022-08-26 14:19:59.025259] -     INFO: Result
                                             | class                |  top1  |  top5  |
                                             -------------------------------------------
                                             | adult                | 74.2 % | 95.5 % |
                                             | lifestyle            | 56.8 % | 80.3 % |
                                             | parents              | 42.7 % | 85.1 % |
                                             | essay                | 40.2 % | 83.9 % |
                                             | humanities           | 57.6 % | 87.0 % |
                                             | religion             | 7.8  % | 33.1 % |
                                             | fantasy_martial_arts | 25.6 % | 83.3 % |
                                             | economic_management  | 28.4 % | 70.3 % |
                                             | magazine             | 64.3 % | 84.8 % |
                                             | travel               | 44.4 % | 82.5 % |
                                             | science              | 13.8 % | 48.7 % |
                                             | romance_bl           | 22.8 % | 70.3 % |
                                             | social               | 70.5 % | 96.7 % |
                                             | novel                | 5.2  % | 41.9 % |    
                                             | children_youth       | 31.3 % | 88.3 % |
                                             | history              | 10.4 % | 64.8 % |
                                             | self_development     | 14.2 % | 57.4 % |
                                             -------------------------------------------
                                             | Average              | 35.9 % | 73.8 % |
    ```
## Detect
  * 학습한 모델로 argument로 준 이미지를 inference하여 결과 출력
    ```shell
    python detect.py \
            --dataset-info=data/bookcover-17.yaml \
            --model-name=efficientnet-b0 \
            --model-path=weights/efficientnet-b0-bookcover.pt \
            --image-path=test.jpg \
            --topk=5
    ```
    * 출력 결과
      ```shell
      [2022-08-26 14:14:54.181588] -     INFO: Load model(efficientnet-b0)
      Loaded pretrained weights for efficientnet-b0
      [2022-08-26 14:15:06.227263] -     INFO: Load image
      [2022-08-26 14:15:06.933078] -     INFO: Start inference
      [2022-08-26 14:15:40.934403] -     INFO: Inference Result
                                                  1 - adult               : 99.51%
                                                 13 - social              : 0.48%
                                                  8 - economic_management : 0.01%
                                                  5 - humanities          : 0.00%
                                                  9 - magazine            : 0.00%
      ```