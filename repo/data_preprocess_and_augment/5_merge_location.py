import pandas as pd

# 1. 두 파일 읽기
location_df = pd.read_csv('./data/merchant_locations.csv', dtype={'zip': str})
train_df = pd.read_csv('./data/augmented_transaction.csv', dtype={'zip': str})

# 2. 병합을 위한 기준 컬럼 설정
merge_keys = ['zip', 'merchant_state', 'merchant_city']

# 3. 중복 제거 (혹시 모를 중복 대비)
location_df_unique = location_df[merge_keys + ['latitude', 'longitude']].drop_duplicates()

# 4. 병합 (left join으로 train_df 기준으로 병합)
merged_df = pd.merge(train_df, location_df_unique, how='left', on=merge_keys)

# 5. 저장
merged_df.to_csv('./raw/test_with_geo.csv', index=False)

print("병합 완료: './data/augmented_transaction_with_geo.csv' 파일로 저장되었습니다.")
