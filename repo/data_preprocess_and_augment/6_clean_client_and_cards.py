import pandas as pd
import numpy as np


def clean_and_process_data(transactions_path, users_path, cards_path):
    # 1. CSV 파일 불러오기
    try:
        transactions_df = pd.read_csv(transactions_path)
        users_df = pd.read_csv(users_path)
        cards_df = pd.read_csv(cards_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the files are in the correct directory.")
        return None, None, None

    # 2. 거래 데이터 전처리
    print("거래 데이터 처리 중...")
    # 'is_online' 피처 추가
    transactions_df['is_online'] = np.where(transactions_df['merchant_city'] == 'ONLINE', 1, 0)

    # 'merchant_region' 피처 추가
    transactions_df['merchant_region'] = transactions_df.apply(
        lambda row: 'ONLINE' if row['is_online'] == 1 else f"{row['merchant_city']}_{row['merchant_state']}",
        axis=1
    )

    # 'zip' 피처를 5자리 문자열로 변환 (0으로 시작하는 경우 처리)
    transactions_df['zip'] = transactions_df['zip'].astype(str).str.zfill(5)
    transactions_df = transactions_df.sort_values(by='date', ascending=True).reset_index(drop=True)

    # 3. 금액 관련 피처를 정제하는 헬퍼 함수
    def clean_dollar_feature(df, column_name):
        if column_name in df.columns:
            df[column_name] = df[column_name].replace(r'[\$,\s]', '', regex=True).astype(float)
        return df

    # 4. 고객 및 카드 데이터 금액 관련 피처 정제
    print("고객 및 카드 데이터 금액 피처 정제 중...")
    money_columns_users = ['per_capita_income', 'yearly_income', 'total_debt']
    for col in money_columns_users:
        users_df = clean_dollar_feature(users_df, col)

    money_columns_cards = ['credit_limit']
    for col in money_columns_cards:
        cards_df = clean_dollar_feature(cards_df, col)

    if 'acct_open_date' in cards_df.columns:
        cards_df['acct_open_date'] = pd.to_datetime(cards_df['acct_open_date'], format='%m/%Y')

    if 'card_on_dark_web' in cards_df.columns:
        cards_df = cards_df.drop(columns=['card_on_dark_web'])

    # 5. 거래 기록이 없는 고객 및 카드 데이터 제거
    print("거래 기록이 없는 고객 데이터 제거 중...")
    # 거래 데이터에 존재하는 고유한 고객 ID 가져오기
    clients_with_transactions = transactions_df['client_id'].unique()

    # 거래 기록이 있는 고객만 남기도록 필터링
    users_df_cleaned = users_df[users_df['id'].isin(clients_with_transactions)].copy()

    # 해당 고객들의 카드 데이터만 남기도록 필터링
    cards_df_cleaned = cards_df[cards_df['client_id'].isin(clients_with_transactions)].copy()

    users_df_cleaned = users_df_cleaned.sort_values(by='id', ascending=True).reset_index(drop=True)
    cards_df_cleaned = cards_df_cleaned.sort_values(by='id', ascending=True).reset_index(drop=True)

    return transactions_df, users_df_cleaned, cards_df_cleaned


if __name__ == '__main__':
    # 예시 파일 경로 (사용 환경에 맞게 수정 필요)
    transactions_file = './data/augmented_transaction_with_geo.csv'
    users_file = './data/users_data.csv'
    cards_file = './data/cards_data.csv'

    transactions_cleaned, users_cleaned, cards_cleaned = clean_and_process_data(
        transactions_file, users_file, cards_file
    )

    if transactions_cleaned is not None:
        print("\n--- 전처리된 거래 데이터 (상위 5개 행) ---")
        print(transactions_cleaned.head())
        print("\n--- 정제된 고객 데이터 (상위 5개 행) ---")
        print(users_cleaned.head())
        print("\n--- 정제된 카드 데이터 (상위 5개 행) ---")
        print(cards_cleaned.head())

        # 6. 정제된 데이터를 새로운 CSV 파일로 저장
        print("\n--- 정제된 데이터 CSV 파일로 저장 중 ---")
        transactions_cleaned.to_csv('./data/cleaned_transactions.csv', index=False)
        users_cleaned.to_csv('./data/cleaned_users.csv', index=False)
        cards_cleaned.to_csv('./data/cleaned_cards.csv', index=False)

        print("정제된 거래 데이터가 'cleaned_transactions.csv' 파일로 저장되었습니다.")
        print("정제된 고객 데이터가 'cleaned_users.csv' 파일로 저장되었습니다.")
        print("정제된 카드 데이터가 'cleaned_cards.csv' 파일로 저장되었습니다.")
