import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 1. 데이터 분할 함수
def split_data_with_fraud_block(df_embeddings, df_meta, train_ratio=0.8, seed=42):
    np.random.seed(seed)

    # 임베딩 데이터와 메타데이터를 인덱스 기준으로 결합
    df_combined = pd.merge(df_meta, df_embeddings, left_index=True, right_index=True, suffixes=('_meta', '_emb'))
    df_combined['seq_index'] = np.arange(len(df_combined))

    # 사기 레이블 컬럼 확인
    if 'last_fraud' not in df_combined.columns:
        print("[WARNING] 'last_fraud' 컬럼이 메타데이터에 없습니다. 'fraud' 컬럼을 사용합니다.")
        df_combined['last_fraud'] = df_combined['fraud_meta']

    train_indices, val_indices = [], []

    # 고객 ID만으로 그룹화하여 분할
    for client_id, group in tqdm(df_combined.groupby('client_id'), desc="데이터 분할 진행", ncols=90):
        group = group.sort_values('window_end_date').reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue

        fraud_arr = group['last_fraud'].values
        fraud_blocks = []
        start_idx = None

        # 사기 거래 블록 식별
        for i, val in enumerate(fraud_arr):
            if val == 1.0 and start_idx is None:
                start_idx = i
            elif val != 1.0 and start_idx is not None:
                fraud_blocks.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            fraud_blocks.append((start_idx, n - 1))

        split_idx = int(n * train_ratio)

        def is_in_block(idx, block):
            return block[0] <= idx <= block[1]

        # 사기 블록 보정
        for block in fraud_blocks:
            if is_in_block(split_idx, block):
                dist_to_start = split_idx - block[0]
                dist_to_end = block[1] - split_idx
                if dist_to_start > dist_to_end:
                    split_idx = block[0]
                else:
                    split_idx = block[1] + 1
                break

        train_indices.extend(group.iloc[:split_idx]['seq_index'].tolist())
        val_indices.extend(group.iloc[split_idx:]['seq_index'].tolist())

    # 인덱스를 사용하여 임베딩 및 메타데이터 분할
    train_embeddings = df_embeddings.iloc[train_indices].reset_index(drop=True)
    val_embeddings = df_embeddings.iloc[val_indices].reset_index(drop=True)
    train_meta = df_meta.iloc[train_indices].reset_index(drop=True)
    val_meta = df_meta.iloc[val_indices].reset_index(drop=True)

    print(f"\n[INFO] 분할 완료. 훈련 데이터 수: {len(train_embeddings)}, 검증 데이터 수: {len(val_embeddings)}")

    return train_embeddings, val_embeddings, train_meta, val_meta


# 2. 메인 실행 로직
if __name__ == "__main__":
    # 파일 경로 설정
    train_ratio = 80
    file_paths = {
        'train_embeddings_input': f'./transformer_train_embeddings.csv',
        'train_meta_input': f'./transformer_train_window_meta.csv',
        'train_embeddings_output': f'./transformer_train_embeddings_{train_ratio}.csv',
        'val_embeddings_output': f'./transformer_train_embeddings_{100 - train_ratio}.csv',
        'train_meta_output': f'./transformer_train_window_meta_{train_ratio}.csv',
        'val_meta_output': f'./transformer_train_window_meta_{100 - train_ratio}.csv'
    }


    print("[STEP 1] 임베딩 및 메타데이터 파일 로드...")
    try:
        df_embeddings = pd.read_csv(file_paths['train_embeddings_input'])
        df_meta = pd.read_csv(file_paths['train_meta_input'])
        df_meta['window_end_date'] = pd.to_datetime(df_meta['window_end_date'])

    except FileNotFoundError as e:
        print(f"[ERROR] 파일이 존재하지 않습니다: {e.filename}. 트랜스포머 모델을 먼저 실행하여 임베딩 파일을 생성해주세요.")
        exit()

    print("[STEP 2] 데이터 분할 시작...")
    train_ratio = train_ratio/100
    train_emb, val_emb, train_meta, val_meta = split_data_with_fraud_block(df_embeddings, df_meta, train_ratio=train_ratio)

    print("\n[STEP 3] 분할된 데이터 저장...")
    train_emb.to_csv(file_paths['train_embeddings_output'], index=False)
    val_emb.to_csv(file_paths['val_embeddings_output'], index=False)
    train_meta.to_csv(file_paths['train_meta_output'], index=False)
    val_meta.to_csv(file_paths['val_meta_output'], index=False)

    print(f"[INFO] 훈련용 임베딩/메타데이터 저장 완료: {file_paths['train_embeddings_output']}, {file_paths['train_meta_output']}")
    print(f"[INFO] 검증용 임베딩/메타데이터 저장 완료: {file_paths['val_embeddings_output']}, {file_paths['val_meta_output']}")