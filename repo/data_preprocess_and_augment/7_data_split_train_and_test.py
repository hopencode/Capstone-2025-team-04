import pandas as pd
import numpy as np

def split_customer_sequence_with_fraud_block(df, train_ratio=0.7, seed=42):
    np.random.seed(seed)

    # datetime 변환
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    train_list = []
    test_list = []

    for client_id, group in df.groupby('client_id'):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue

        # fraud 연속 구간 구하기
        fraud_arr = group['fraud'].values
        fraud_blocks = []
        start_idx = None
        for i, val in enumerate(fraud_arr):
            if val == 1.0 and start_idx is None:
                start_idx = i
            elif val != 1.0 and start_idx is not None:
                fraud_blocks.append((start_idx, i-1))
                start_idx = None
        if start_idx is not None:
            fraud_blocks.append((start_idx, n-1))

        # 초기 split 위치
        split_idx = int(n * train_ratio)

        # split_idx가 fraud 블록 중간에 걸리면 블록 전체를 한쪽에 포함하도록 조정
        def is_in_block(idx, block):
            return block[0] <= idx <= block[1]

        for block in fraud_blocks:
            if is_in_block(split_idx, block):
                dist_to_start = split_idx - block[0]
                dist_to_end = block[1] - split_idx
                if dist_to_start > dist_to_end:
                    # train에 몰기 (split_idx를 fraud block 시작으로 조정)
                    split_idx = block[0]
                else:
                    # test에 몰기 (split_idx를 fraud block 끝 다음 인덱스로 조정)
                    split_idx = block[1] + 1
                break

        # 분할
        train_part = group.iloc[:split_idx].copy()
        test_part = group.iloc[split_idx:].copy()

        train_list.append(train_part)
        test_list.append(test_part)

    # 최종 concat 후 인덱스 초기화
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv('./data/cleaned_transactions.csv', dtype={'zip': str})
    df['date'] = pd.to_datetime(df['date'])

    train_df, test_df = split_customer_sequence_with_fraud_block(df, train_ratio=0.8)

    # 학습용, 테스트용 CSV 파일로 저장
    train_df.to_csv('./data/train_transactions_Clean.csv', index=False)
    test_df.to_csv('./data/test_transactions_Clean.csv', index=False)

    # 간단 출력 확인
    # print(f"Train size: {len(train_df)}, Fraud in train: {train_df['fraud'].sum()}")
    # print(f"Test size: {len(test_df)}, Fraud in test: {test_df['fraud'].sum()}")
