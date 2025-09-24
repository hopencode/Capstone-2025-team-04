import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
from tqdm import tqdm

### 노이즈 범위
DATE_NOISE_MIN, DATE_NOISE_MAX = -30, 30   # date 노이즈: -30~+30분
AMOUNT_NOISE_RATE = 0.1                    # amount 노이즈: ±10%

def add_time_noise(ts, min_min=DATE_NOISE_MIN, max_min=DATE_NOISE_MAX):
    # datetime 내 랜덤 minutes 시프트
    delta = np.random.randint(min_min, max_min + 1)
    return ts + pd.Timedelta(minutes=delta)

def add_amount_noise(val, rate=AMOUNT_NOISE_RATE):
    # amount에 ±노이즈 비율 적용
    noise = np.random.uniform(-rate, rate)
    newval = max(0, val + val * noise)
    return round(newval, 2)

print("1. 데이터 로딩 및 정합성 처리")
start_time = time.time()
df = pd.read_csv('./raw/transactions_fraud_label_preprocess.csv', dtype={'zip': str, 'mcc': str})

print(f"   > 데이터 전체 로딩 완료 ({time.time() - start_time:.2f}초, {len(df):,}건)\n")
df['zip'] = df['zip'].apply(lambda x: str(x).zfill(5) if x.isdigit() else str(x).strip())
df['mcc'] = df['mcc'].apply(lambda x: str(x).zfill(4) if x.isdigit() else str(x).strip())
df['date'] = pd.to_datetime(df['date'])
#if 'mcc_type' in df.columns:
#    df = df.drop(columns=['mcc_type'])
print("   > 컬럼 정규화·필요없는 컬럼 제거 완료\n")

zip_combo_set = set(df[['zip','merchant_state','merchant_city']].itertuples(index=False, name=None))
client_card_set = set(df[['client_id','card_id']].itertuples(index=False, name=None))
valid_mccs = set(df['mcc'])

### 2. 사기 거래 증강 (슬라이딩 윈도우 + 변형)
print("2. 사기 거래 슬라이딩 윈도우 증강/변형 시작")
WINDOW_SIZE, STRIDE = 5, 1
fraud_df = df[df['fraud'] == 1].copy()
orig_fraud_count = len(fraud_df)

grouped = list(df.groupby(['client_id', 'card_id']))
augmented = []

for g_idx, ((client_id, card_id), group) in enumerate(tqdm(grouped, desc="그룹별 윈도우 슬라이싱"), 1):
    group = group.reset_index(drop=True)
    if (client_id, card_id) not in client_card_set:
        continue
    for start in range(0, len(group)-WINDOW_SIZE+1, STRIDE):
        window = group.iloc[start:start+WINDOW_SIZE].copy()
        if window['fraud'].sum() > 1:
            valid_flag = True
            for _, row in window.iterrows():
                if (row['zip'], row['merchant_state'], row['merchant_city']) not in zip_combo_set or row['mcc'] not in valid_mccs:
                    valid_flag = False
                    break
            if valid_flag:
                for i, row in window.iterrows():
                    if row['fraud'] == 1:
                        new_row = row.copy()
                        # amount 노이즈
                        new_row['amount'] = add_amount_noise(row['amount'])
                        # date 노이즈
                        new_row['date'] = add_time_noise(row['date'])
                        # use_chip 변형 없이 그대로 유지 (온라인 거래는 반드시 Online Transaction, 아닌 경우 chip/swipe만)
                        augmented.append(new_row)

aug_fraud_df = pd.DataFrame(augmented)
aug_fraud_df = pd.concat([fraud_df, aug_fraud_df], ignore_index=True).drop_duplicates()
print(f"   > 변형포함 증강-최종 사기 거래 수: {len(aug_fraud_df):,}")

# 4~8배 미만 시 추가 복제
target_fraud = max(orig_fraud_count * 4, len(aug_fraud_df))
if len(aug_fraud_df) < target_fraud:
    extra = aug_fraud_df.sample(target_fraud - len(aug_fraud_df), replace=True, random_state=42)
    aug_fraud_df = pd.concat([aug_fraud_df, extra]).reset_index(drop=True)
print(f"   > 증강-최종 사기 거래 수(최종): {len(aug_fraud_df):,}\n")

### 3. 정상 거래 KNN 유사 기반 언더샘플링
print("3. 정상 거래 KNN 유사 기반 언더샘플링")
non_fraud_df = df[df['fraud'] == 0].copy()

zip_le = LabelEncoder().fit(df['zip'])
mcc_le = LabelEncoder().fit(df['mcc'])
aug_fraud_df['zip_enc'] = zip_le.transform(aug_fraud_df['zip'])
aug_fraud_df['mcc_enc'] = mcc_le.transform(aug_fraud_df['mcc'])
non_fraud_df['zip_enc'] = zip_le.transform(non_fraud_df['zip'])
non_fraud_df['mcc_enc'] = mcc_le.transform(non_fraud_df['mcc'])

scaler_amount = MinMaxScaler()
aug_fraud_df['amount_scl'] = scaler_amount.fit_transform(aug_fraud_df[['amount']])
non_fraud_df['amount_scl'] = scaler_amount.transform(non_fraud_df[['amount']])

scaler_zip = MinMaxScaler()
aug_fraud_df['zip_scl'] = scaler_zip.fit_transform(aug_fraud_df[['zip_enc']])
non_fraud_df['zip_scl'] = scaler_zip.transform(non_fraud_df[['zip_enc']])

scaler_mcc = MinMaxScaler()
aug_fraud_df['mcc_scl'] = scaler_mcc.fit_transform(aug_fraud_df[['mcc_enc']])
non_fraud_df['mcc_scl'] = scaler_mcc.transform(non_fraud_df[['mcc_enc']])

aug_fraud_df['mcc_wgt'] = aug_fraud_df['mcc_scl'] * 0.1
non_fraud_df['mcc_wgt'] = non_fraud_df['mcc_scl'] * 0.1

feat_cols = ['amount_scl', 'zip_scl', 'mcc_wgt']

X_fraud = aug_fraud_df[feat_cols].values
X_normal = non_fraud_df[feat_cols].values

print(f"   > KNN - 정상({len(X_normal):,}), 사기({len(X_fraud):,}) 거래 fitting 시작")
neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
neighbors.fit(X_normal)
print(f"   > KNN learning 완료")
print("   > 사기 거래에 대해 최근접 정상거래 검색 시작")
distances, indices = neighbors.kneighbors(X_fraud)
print("   > KNN 검색 완료")

# 중복 없는 유사 샘플 우선 추출 (필수 목표 개수(n_normal)만큼 보장)
indices_set = set(indices.flatten())
n_normal = len(aug_fraud_df)*9
sel_normal_df = non_fraud_df.iloc[list(indices_set)].copy()
if len(sel_normal_df) < n_normal:
    print(f"   > KNN 유사 샘플만으로 부족: {len(sel_normal_df):,}개. {n_normal - len(sel_normal_df):,}개 추가 랜덤 추출")
    rest = non_fraud_df.drop(sel_normal_df.index)
    extra_needed = n_normal - len(sel_normal_df)
    if extra_needed > len(rest):
        print("   > 경고: 전체 정상 거래 수가 목표치보다 적어, 가능한 만큼만 추가")
        extra_needed = len(rest)
    sel_normal_df = pd.concat([sel_normal_df, rest.sample(extra_needed, random_state=42)])

print(f"   > 최종 유사+랜덤 정상 거래 추출 건수: {len(sel_normal_df):,}\n")

### 4. id 신규 부여(순번)
aug_fraud_df = aug_fraud_df.copy()
sel_normal_df = sel_normal_df.copy()
aug_fraud_df['id'] = np.arange(1, len(aug_fraud_df)+1)
sel_normal_df['id'] = np.arange(len(aug_fraud_df)+1, len(aug_fraud_df)+len(sel_normal_df)+1)

### 5. 컬럼 순서 및 저장
use_cols = [
    'id','date','client_id','card_id','amount','use_chip','merchant_id','merchant_city',
    'merchant_state','zip','mcc','errors','mcc_type','fraud'
]
final_df = pd.concat([aug_fraud_df, sel_normal_df], ignore_index=True)[use_cols]
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df = final_df.sort_values(by='date', ascending=True).reset_index(drop=True)

print(f'최종 저장 샘플 수: {len(final_df):,}')
final_df.to_csv('./data/augmented_transaction.csv', index=False)
print('완료!')

