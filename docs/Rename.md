# Rename dataset
* class 이름이 한글일 경우 유니코드 에러가 발생할 수 있으므로 class 폴더를 영문명으로 변경 및 이미지 이름을 숫자 이름으로 변경하는 툴
* 원본 데이터셋 구조는 [링크](#Dataset 구조)의 원본 데이터셋 구조와 같아야 한다.
## Dataset 구조
### 원본 데이터셋 구조
```shell
dataset
  ├─ 라이프스타일
  │   ├─ 책이름1.jpg
  │   ├─ 책이름2.jpg
  │   ...
  │   └─ 책이름n.jpg
  ├─ 만화
  ...
  ├─ 좋은부모
  └─ 청소년
```

## Data 정보 파일
* 한글 Class 이름을 영어로 매핑하기 위해 아래 항목을 작성한 data.yaml 파일이 필요하다.
* 예시 파일 ```data/bookcover_class.yaml```
```yaml
nc: 38
eng_class: ["Adult", "Health_Hobbies-Leisure", ...]
kor_class: ["19금", "건강,취미,레저", ...]
```
## 사용 방법
* 위 설명과 같이 원본 데이터셋의 형식에 맞춰 데이터셋을 준비하고 ```data/bookcover_class.yaml```과 같이 data 정보 파일을 준비한다. 
### rename_dataset.py
```shell
python rename_dataset.py \
    --origin-dir=/origin \
    --target-dir=/dataset/bookcover \
    --dataset-ratio=1,1,4 \
    --dataset-class-info=data/bookcover_class.yaml \
    --is-filter \
    --min-cls-img-nb=1000 \
    --save-dataset-info \
    --debug
```
#### Arguments
* ```--origin-dir```: 원본데이터 디렉토리 경로
* ```--target-dir```: rename 후 저장할 디렉토리 경로(```target_dir```은 비어있어야 함)
* ```--dataset-ratio```: test, val, train 데이터셋의 비율(defalt: 1,1,4)
* ```--dataset-class-info```: Data 정보 파일 경로
* ```--is-filter```: ```min-cls-img``` 인자 값을 참조하여 class를 필터링 할지 결정 
* ```--min-cls-img```: class에 포함된 최소 이미지의 수(해당 인자보다 class에 포함된 이미지가 적을 경우 생성한 데이터셋에 해당 class가 포함되지 않음
* ```--save-dataset-info```: dataset 매핑 정보를 저장할지 결정(데이터셋 확인용)
* ```--debug```: 처리 중간에 출력구문을 출력할지 말지 결정

### Rename 이후 데이터셋 구조
```shell
renamed_dataset
  ├─ Lifestyle
  │   ├─ 000001.jpg
  │   ├─ 000002.jpg
  │   ...
  │   └─ 00000n.jpg
  ├─ Comic
  ...
  ├─ Good_Parents
  └─ Youth
```