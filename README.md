# Melon Playlist Continuation: Hunchbrown - KNN 기반의 앙상블 방법
카카오 아레나 Melon Playlist Continuation 과제를 해결하기 위해서 KNN 기반의 방법을 중심으로 여타의 다른 방법을 결합하였다. 총 세 개의 방법을 일정한 가중치에 따라서 결합하였으며, Cold Start 문제를 해결하기 위해서 Playlist 타이틀 정보를 활용한 방법을 추가하였다.

## 데이터 사용

res 폴더 안에 아레나에서 다운받은 파일 train.json, val.json, test.json이 있다고 가정한다. Matrix Factorizatioin 방법을 사용하기 위해서 검증시에는 train.json과 val.json을 최종 제출시에는 train.json과 test.json을 사용한다. 최종 추천 결과는 arena_data 아래에 results 폴더의 results.json으로 저장된다.

추천 하는 태그 및 노래는 train.json에 등장하는 태그와 노래로 한정힌다. 생성하는 Playlist-Item(Tag/Song) Matrix의 크기를 제한하며 동시에 val.json이나 test.json에서만 등장하는 태그 또는 노래는 사실상 중요도가 떨어진다고 판단하였다. 


## 추천 결과 생성

추천 결과 생성은 별도의 신경망 모델을 학습할 필요 없이 바로 생성한다. 

데이터(train.json, val.json, test.json)를 res 폴더 아래에 위치시키고 arena_data 빈폴더를 준비한다. arena_data 폴더 아래 results 폴더에 추천 결과가 생성된다. 

추천 모델 생성을 위해서는 requirements.txt에 사용하는 외부 라이브러리를 pip을 통해 설치하고 카카오 형태소 분석기 khaiii를 빌드해야한다. 별도의 기분석 사전이 없으므로 [khaiii 공식 빌드](https://github.com/kakao/khaiii/wiki/%EB%B9%8C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98) 방법을 사용한다. 

사전 환경 설정 이후 inference.py를 실행한다.

```bash
$> python inference.py
```

inference.py는 argparse를 통해 command line arguments를 받을 수 있으며, 기본값은 학습을 위한 `./res/train.json`과 테스트를 위한 `./res/test.json`이다. 검증시에는 학습과 테스트를 위한 파일명을 변경한다.

```bash
$> python inference.py --train_fname=./res/train.json --test_fname=./res/val.json
```

빠른 implicit 계산을 위해서 **GPU 환경**에서 실행하는 것을 권장한다.


## 추천 결과 생성 방법

### 1. IDF KNN 방법

추천 모델의 핵심이 되는 방법이다. 생성된 학습용 Playlist-Item(Tag/Song) Matrix와 테스트용 Playlist-Item(Tag/Song) Matrix 사이의 코사인 유사도를 계산한다. Playlist의 유사도를 계산할 때 노래와 태그 정보를 모두 사용하며 특히 노래에 많은 가중치를 줌으로써 노래가 유사한 Playlsit가 유사한 Playlsit로 간주한다. 계산된 코사인 유사도에 근거하여서 테스트용 Playlist와 가장 유사한 K개의 Playlist를 학습용 데이터에서 찾는다. K개의 학습용 Playlist의 Item(Tag/Song) 벡터를 합한 것을 테스트용 Playlist의 Item(Tag/Song) 벡터이자 각 Item에 대한 Rating으로 간주한다.

Playlist를 Item(Tag/Song)의 벡터로 표현하는 과정에서 Tf-Idf 변환을 사용한다. Tf-Idf 변환을 통해서 학습용 Playlist에서 적게 나온 Item(Tag/Song)의 중요도를 높인다. 검증용 데이터로 실험한 결과 Playlist-Item Matirx에 Tf-IDf 변환을 적용한 IDF KNN 방법이 적용하지 않은 것보다 좋은 성능을 보였다.

KNN의 하이퍼파라미터는 학습용 데이터에 따른 비율로 설정한다. 현재 학습용 데이터의 0.001 비율로 K를 정한다. 추가로 10의 배수로 K개가 떨어지도록 보정한 결과 K는 120으로 설정한다. 높은 비율(0.003)에 비해서 0.001 비율이 높은 성능을 보이므로 이를 채택한다.

### 2. ALS Matrix Factorization 방법

Matrix Factorization은 추천 모델을 구현하는 데 있어서 전형적인 방법이다. Matrix Factorization을 통해 Plyalsit-Item(Tag/Song)의 비어 있는 값(rating)을 추론한다. 이를 위해서 외부 라이브러리 implicirt을 활용한다.

Matrix Factorization을 위해서 학습 데이터(train.json)과 테스트 데이터(val.json or test.json)에서 각각 만든 Playlist-Item(Tag/Song) Matrix를 열 방향으로 쌓는다. 이렇게 쌓인 Matrix를 Factorization 하면서 비어 있는 값을 추론한다.

노래의 rating을 추론하는 데 있어서는 전통적인 Matrix Factorization을 그대로 활용하지만 태그를 추천하는데 있어서는 다소 변형이 필요하다. 노래에 비해서 태그는 Playlist와의 연관관계가 낮으므로 Matrix Factorization 자체만으로는 Rating을 추론하기 어려울 것이라 판단하였다. 따라서 태그를 추론할 때에는 Playlist 유사도 정보를 바탕으로 학습용 Playlsit와 가장 유사한 100개의 태그 벡터를 선정하여서 이를 합한 것으로 사용한다.

### 3. Item CF 방법

테스트 Playlist에 있는 시드(seed) 아이템(Tag/Song)을 기준으로 Rating을 추론한다. 유사한 아이템을 찾는 Collaborative Filtering 방법이다.

Playlist-Item(Tag/Song) Matrix를 전치하여서 Item에 대한 Playlist 벡터를 만들고 이를 활용하여서 Item간의 코사인 유사도를 구한다. 구해진 태그 유사도 Matirx와 노래 유사도 Matrix를 사용해서 테스트 Playlsit에 있는 시드 아이템과 유사한 아이템을 선택한다.

IDF 방식과 마찬가지로 Tf-Idf 변환을 거친다.

### 4. Title 방법 - For Cold Start

Cold Start 문제는 시드(seed) 아이템(Tag/Song)이 없는 상황으로 이러한 경우 Collaborative Filtering 방식으로는 추천이 이루어지기 어렵다. 이때 Playlist Title 정보를 바탕으로 유사한 Playlist를 찾고 추천한다.

Plyalist 타이틀을 카카오 한글 형태소 분석기 khaiii를 사용해서 분석한다. 이때 명사, 동사와 같이 문법적으로 유의미한 단어만을 가지고 Title-Item(Tag/Song) Matrix를 구축한다. 시드가 없는 테스트 데이터에 대해서 그 타이틀을 형태소 분석하고 유의미한 단어를 선정한 이후 Matirx에서 해당 단어의 벡터를 찾아서 합한다.

3가지 기본 방법과 시드가 없는 데이터에 대한 추가 방법을 적절한 가중치와 함께 최종 판단을 내린다. 가중치는 train.json과 val.json을 사용해서 최적의 상태를 실험적으로 얻었다. 전체적으로 IDF KNN이 태그와 노래에 있어서 가장 좋은 성능을 보였으며 그 다음으로 MF와 Item CF 방식이 뒤를 이었다. Title 방법은 추천 모델의 성능을 향상 시키는 데 영향을 주었다. 