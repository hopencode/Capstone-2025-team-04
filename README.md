## 시계열 특성을 활용한 다양한 이상 거래 탐지 딥러닝 모델 개발과 성능 분석

### 1. 프로젝트 배경
#### 1.1. 금융 산업 현황 및 기존 문제점
금융 산업에서 사기 거래 탐지는 카드 해킹, 무단 도용, 보이스피싱 등 다양한 금융 범죄를 조기에 발견하고 예방하는 데 핵심적인 역할을 한다.  
최근 인터넷 뱅킹, 모바일 결제, 온라인 커머스 등 디지털 금융 서비스의 확산으로 거래 데이터의 규모가 폭발적으로 증가하고 있으며, 이에 따라 금융 사기의 발생 빈도와 규모도 꾸준히 확대되고 있다.  

이러한 상황에서 머신러닝이 사기 탐지에 도입되면서 XGBoost, 랜덤 포레스트 등 결정트리 기반 모델이 다양한 전처리 기법과 결합하여 일정 수준 이상의 성과를 보였다.  
그러나 기존 접근법에는 다음과 같은 한계가 존재한다.  
- 거래 데이터의 **시계열적 특성**과 **장기 의존성**을 충분히 반영하지 못함  
- 거래 금액, 빈도 등 단일 시점의 정적 특징만으로는 복잡한 패턴이나 점진적 변화를 탐지하기 어려움  
- 고객·카드 특성과 같은 **부가 정보**를 충분히 활용하지 못해 탐지 정확도 및 일반화 성능에 제약이 있음  

#### 1.2. 필요성과 기대효과
최근 딥러닝은 이러한 한계를 극복하기 위한 대안으로 주목받고 있다. 특히 **트랜스포머(Transformer)** 모델은 어텐션 메커니즘을 활용하여 시계열 데이터 내 모든 시점 간 관계를 병렬적으로 학습할 수 있어, 거래 데이터 분석에서 복잡하고 장기적인 시간 패턴을 효과적으로 포착할 수 있다.  

본 프로젝트는 트랜스포머 기반 임베딩을 활용하여 거래 데이터의 시계열적 특성을 학습하고, 고객 및 카드와 같은 다양한 속성을 결합하여 이상 거래 탐지 성능을 개선하는 것을 목표로 한다. 이를 통해 기존 머신러닝 기반 접근법의 한계를 보완하고, 실제 금융 보안 시스템에서 활용 가능한 성능을 확보할 수 있을 것으로 기대된다.  


### 2. 개발 목표
#### 2.1. 목표 및 세부 내용
본 프로젝트의 핵심 목표는 거래 데이터의 시계열 정보와 부가 정보를 종합적으로 활용하여 **이상 거래 탐지**에 적합한 모델을 찾는 것이다. 이를 위해 다음과 같은 세부 연구 방향을 설정하였다.

1. **트랜스포머 기반 시계열 임베딩 적용**  
   - 거래 데이터의 시간적 흐름과 장기 의존성을 효과적으로 학습  
   - 전통적 머신러닝 기법에서 간과되기 쉬운 복잡한 패턴 포착  

2. **속성 데이터 융합을 통한 탐지 모델 개발**  
   - 트랜스포머 임베딩과 고객·카드 속성을 통합하여 다양한 모델 구현  
   - 비교 모델  
     - **MLP**: 거래 및 고객·카드 정보를 결합한 단순 신경망  
     - **TGN**: 시간 정보를 반영한 그래프 신경망  
     - **HTGN**: 고객, 카드, 거래 등 이종 데이터를 반영한 그래프 신경망  

3. **성능 검증 및 비교 분석**  
   - 평가 지표: Accuracy, Precision, Recall, F1-score, PR-AUC  
   - 단일 트랜스포머와 MLP, TGN, HTGN 모델의 성능을 비교하여 최적의 전략 도출  

#### 2.2. 기존 서비스 대비 차별성
- 단일 거래 기반 분석이 아닌 **시계열적 맥락을 반영**  
- 고객 및 카드 속성과 같은 **부가 정보를 융합**하여 다차원적 분석 수행  
- **다층 퍼셉트론(Multi-Layer Perceptron)**, **이종 그래프 기반 모델(HTGN)** 등을 통해 기존 연구 대비 더 정교한 관계 학습 가능  

#### 2.3. 사회적 가치 및 기대 효과
- 금융 사기 조기 탐지를 통해 **피해 규모 감소** 및 **금융 보안 강화**  
- 대규모 거래 데이터 분석 기술 고도화를 통해 **산업 현장 적용 가능성 확대**

### 3. 시스템 설계
#### 3.1. 시스템 구성도
본 프로젝트는 거래 데이터의 시계열 특성과 다양한 부가 정보를 종합적으로 활용한 이상 거래 탐지 모델 개발을 목표로 합니다. 연구 과정은 다음과 같이 구성됩니다.

프로젝트 전체 단계 흐름:
```
Raw Transaction Data → 데이터 전처리 → 사기 거래 증강 & 정상 거래 언더샘플링 → 추가 전처리
→ 트랜스포머 임베딩 생성 → MLP / TGN / HTGN 모델 학습 → 성능 평가 및 비교
```

1. **데이터 전처리**: 사기 거래 라벨링, 거래 데이터에서 결측치 처리 
2. **사기 거래 증강 및 정상 거래 언더샘플링**: 슬라이딩 윈도우를 활용한 사기 거래 증강과, 정상 거래 언더샘플링  
3. **추가 전처리**:  상점 좌표 연결 등 추가 전처리 및 학습/테스트용 데이터 분할
4. **트랜스포머 기반 임베딩 생성**: 트랜스포머 입력용 시퀀스 생성 및 거래 시퀀스 간 관계를 학습하여 임베딩 벡터 추출  
5. **모델별 학습**:
   - **MLP**: 거래, 고객, 카드 정보를 통합한 단순 신경망 기반 탐지 모델  
   - **TGN**: 거래 간 시간 정보를 반영한 그래프 신경망 기반 탐지 모델  
   - **HTGN**: 고객, 카드, 거래 등 이종 데이터를 반영한 그래프 기반 탐지 모델  
6. **모델별 성능 평가 및 비교**: PR-AUC, F1-score 등 지표를 통해 모델별 성능을 평가 및 비교



#### 3.2. 사용 기술 및 환경
##### 공통 사항
- **개발 언어**: Python 3.10 이상
- **개발 환경**: GPU 사용 권장 (트랜스포머 및 HTGN 모델 기준 5060ti GPU의 PC에서 개발)

##### MLP 개발 환경
- **프레임워크**: PyTorch==2.8.0
- **데이터 처리 라이브러리**
  - Pandas==2.2.3
  - NumPy==1.26.4
  - Scikit-learn==1.2.2
  - xgboost==3.0.4
  - imbalanced-learn==0.12.3
 - **시각화 및 유틸리티**
 - matplotlib==3.9.0
 - seaborn==0.13.2
 - tqdm
 - joblib

##### TGN 개발 환경
- **프레임워크**: PyTorch==2.8.0
- **TGN 모델 코드**: [TGN Official Repository](https://github.com/twitter-research/tgn) 에서 클론하여 사용
- **데이터 처리 라이브러리**
  - Pandas==2.0.2
  - NumPy==1.23.4
  - Scikit-learn==1.2.1
- **기타 라이브러리**
  - matplotlib
  - tqdm

##### HTGN 개발 환경
- **프레임워크**: PyTorch==2.8.0
- **그래프 신경망 라이브러리**: PyTorch Geometric (버전은 설치된 PyTorch에 맞춤)
- **추가 종속 라이브러리**: pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv
- **데이터 처리 라이브러리**
  - Pandas==2.0.1
  - NumPy==1.23.3
  - Scikit-learn==1.2.0


### 4. 개발 결과
#### 4.1. 전체 시스템 흐름도
<img width="431" height="606" alt="Image" src="https://github.com/user-attachments/assets/303d99c5-2bc5-4074-bd28-088f81c05e29" />

#### 4.2. 기능 설명 및 모델별 명세
| 단계/기능             | 입력                          | 출력                    | 설명                                                                                     |
| ------------------- | ---------------------------- | --------------------- | -------------------------------------------------------------------------------------- |
| 데이터 전처리          | 원시 거래 데이터 csv 파일                | 전처리된 거래 데이터 csv 파일       | 사기 라벨링, 결측치 처리  |
| 사기 거래 증강 및 정상 거래 언더샘플링 | 전처리된 거래 데이터 csv 파일             | 증강/언더샘플링된 데이터 csv 파일    | 슬라이딩 윈도우 기반 사기 거래 증강, 유사 정상 거래 우선 선택 및 추가 랜덤 언더샘플링                   |
| 추가 전처리            | 증강/언더샘플링 데이터 csv 파일           | 추가 전처리 후 학습/테스트용으로 분할된 데이터 각각 csv 파일  | 상점 좌표 등 추가 피처 결합, 학습/테스트 데이터셋 분리                                        |
| 트랜스포머 임베딩 생성    | 학습/테스트용으로 분할된 데이터 csv 파일 <br>+ 카드 데이터 csv 파일   | 거래 시퀀스 임베딩        | 카드 데이터 피처 활용한 추가 피처 생성 + 시퀀스 내 모든 거래 간 관계 학습, 임베딩 벡터 생성                                           |
| MLP 학습              | 트랜스포머 임베딩 (학습용/검증용/테스트용 분할) 각 csv 파일 <br>+ 고객/카드 데이터 각 csv 파일 | 이상 거래 분류 결과 출력      | 단순 신경망 기반 분류, 트랜스포머 임베딩과 고객/카드 정보를 결합하여 이상 거래 예측                     |
| TGN 학습              | 트랜스포머 임베딩 (학습용/검증용/테스트용 분할) 각 csv 파일 <br>+ 고객/카드 데이터 각 csv 파일 | 이상 거래 분류 결과 출력        | 거래 간 시간 정보와 엣지 피처를 반영한 시계열 그래프 구조로 고객·카드의 과거 맥락을 노드 임베딩에 통합                         |
| HTGN 학습             | 트랜스포머 임베딩 (학습용/테스트용 분할) 각 csv 파일 <br>+ 고객/카드 데이터 각 csv 파일 | 이상 거래 분류 결과 출력       | 고객, 카드, 거래 등 이종 노드를 포함한 그래프 신경망 구성, 거래 노드간 엣지를 통해 직접적인 시계열 특성 반영 |

#### 4.3. 모델 성능 결과 비교
- PR 커브
<img width="872" height="722" alt="Image" src="https://github.com/user-attachments/assets/47de54c0-e5ea-43ff-8c98-a29927551cb9" />
<br>
<br>

- Fraud & Macro F1-score 비교
<img width="1377" height="964" alt="Image" src="https://github.com/user-attachments/assets/d5ad37b1-3494-4b43-ba87-48df701ccce8" />

#### 4.4. 디렉토리 구조
```
repo/
│
├── data_preprocess_and_augment/    # 데이터 전처리 및 증강
│   ├── data/                       # 원본 및 전처리 결과 저장
│   └── requirements.txt
│
├── transformer/                    # 거래 데이터 시계열 정보 처리 Transformer 모델
│   ├── src/                        # 모델 코드
│   ├── data/                       # 실행용 데이터
│   ├── trained_model/
│   ├── embedding_result/
│   ├── npz/
│   ├── scaler_and_encoder/
│   └── requirements.txt
│
├── mlp/                            # 이상 거래 탐지 MLP 모델
│   ├── src/
│   ├── data/
│   ├── notebooks/                  # 개발 및 실험 노트북
│   ├── trained_model/
│   ├── results/
│   ├── requirements.txt
│   └── run_mlp.sh
│
├── tgn/                            # 이상 거래 탐지 TGN 모델
│   ├── src/
│   ├── data/
│   ├── trained_model/
│   └── results/
│
├── htgn/                           # 이상 거래 탐지 HTGN 모델
│   ├── src/
│   ├── data/
│   ├── embedding_result/
│   ├── scaler_and_encoder/
│   ├── trained_model/
│   ├── npz/
│   └── requirements.txt
│
└── README.md
```

### 5. 설치 및 실행 방법
- 수집 데이터 링크 [Financial Transactions Dataset: Analytics](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) 
  - 데이터 용량 초과로 저장소 업로드 불가로 링크 첨부
  - 이로 인해 저장소에 업로드하지 못한 입출력 파일이 있으므로 디렉토리 구조 및 코드 상의 입출력 파일 확인 필요
- MLP, TGN은 트랜스포머 임베딩 이후 학습용 임베딩 데이터에 대해 추가로 MLP, TGN 모델 학습을 위한 학습용/검증용 데이터 분리 코드 실행 필요
  - HTGN은 코드 내부에서 입력받은 학습용 임베딩을 학습용/검증용으로 분할해서 활용
- 트랜스포머, HTGN 모델 개발 환경은 5060ti로 비슷한 수준 이상의 GPU 사용을 권장

#### 5.1. 데이터 전처리 및 Transformer 설치 및 실행

- **주요 라이브러리 설치**
```bash
pip install numpy==1.23.5
pip install pandas==2.0.3
pip install scikit-learn==1.2.2
pip install geopy==2.4.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

- **실행**
```bash
# data_preprocess_and_augment 디렉토리
python 1_transaction_data_fraud_labeling.py
python 2_preprocessing_missing_values.py
python 3_transaction_data_augment.py
python 4_get_location.py
python 5_merge_location.py
python 6_clean_client_and_cards.py
python 7_data_split_train_and_test.py

# transformer/src 디렉토리
python transformer_transaction_embedding.py
```

#### 5.2. MLP 설치 및 실행
- **가상 환경 생성 및 활성화 (Conda or venv)** 
```bash
#Conda
conda create -n mlp_env python=3.10 
conda activate mlp_env

#venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```
- **주요 라이브러리 설치**
```bash
pip install -r requirements.txt
```
- **Shell script 전체 파이프라인 실행**
```bash
chmod +x run_mlp.sh
./run_mlp.sh
```
- **개별 단계 직접 실행**
```bash
python -m src.main --mode train # 모델 학습 및 평가 실행
python -m src.main --mode analyze # 저장된 결과 분석 및 시각화
```

#### 5.3. TGN 설치 및 실행
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas==2.0.2 numpy==1.23.4 scikit-learn==1.2.1
pip install matplotlib tqdm
# TGN 원본 저장소 클론
git clone https://github.com/twitter-research/tgn.git
```

#### 5.4. HTGN 설치 및 실행
- **주요 라이브러리 설치**
```bash
pip install numpy==1.23.5
pip install pandas==2.0.3
pip install scikit-learn==1.2.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch_geometric
```

- **실행**
```bash
# htgn/src 디렉토리
python htgn_fraud_detect.py
```

### 6. 소개 자료 및 영상
#### 6.1. 프로젝트 소개 자료
[프로젝트 소개 자료](https://github.com/pnucse-capstone2025/Capstone-2025-team-04/blob/main/docs/03.%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%8C%8004_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf)
#### 6.2. 프로젝트 소개 영상
[프로젝트 소개 영상](https://youtu.be/2UTNNKXFWw0?si=PE7ZSIX1KHIAE798)

### 7. 팀 구성
#### 7.1. 팀원별 소개 및 역할 분담

| 이름 | 역할 |
|:---:|:---|
| 배근호 | 데이터 전처리<br>거래 데이터 시계열 정보 처리를 위한 트랜스포머 모델 개발 및 평가<br>이상 거래 탐지 HTGN 모델 개발 및 평가 |
| 추민 | 데이터 전처리<br>이상 거래 탐지 MLP 모델 개발 및 평가 |
| 윤소현 | 데이터 전처리<br>이상 거래 탐지 TGN 모델 개발 및 평가 |


### 8. 참고 문헌 및 출처
- 트랜스포머
  [파이토치 트랜스포머 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

- TGN
  TGN 모델은 원 논문 및 공식 구현 코드를 기반으로 하였다.
  ```
  @inproceedings{tgn_icml_grl2020,
      title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
      author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico 
      Monti and Michael Bronstein},
      booktitle={ICML 2020 Workshop on Graph Representation Learning},
      year={2020}
  }
  ```
  GitHub: https://github.com/twitter-research/tgn

- HTGN
  이종 그래프 구성을 위한 [PyG 문서](https://pytorch-geometric.readthedocs.io/en/2.6.0/index.html#)
  - [HeteroData](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)
  - [HGTConv](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.HGTConv.html#torch_geometric.nn.conv.HGTConv)
