import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CustomerCardFeatureEngineering:
    """고객 및 카드 데이터를 활용한 고급 특징 생성"""

    def __init__(self, users_df, cards_df):
        self.users_df = users_df
        self.cards_df = cards_df
        self.scaler = StandardScaler()

    def create_integrated_features(self, meta_df):
        """통합 특징 생성 메인 함수"""

        # 시간 특징 먼저 추가
        meta_df = meta_df.copy()
        meta_df['window_end_date'] = pd.to_datetime(meta_df['window_end_date'])
        meta_df['hour'] = meta_df['window_end_date'].dt.hour
        meta_df['day_of_week'] = meta_df['window_end_date'].dt.dayofweek
        meta_df['month'] = meta_df['window_end_date'].dt.month
        meta_df['is_weekend'] = meta_df['day_of_week'].isin([5, 6]).astype(int)
        meta_df['is_night'] = meta_df['hour'].between(22, 6).astype(int)
        meta_df['is_business_hour'] = meta_df['hour'].between(9, 17).astype(int)

        # 1. 고객 프로필 특징
        customer_features = self._create_customer_profile_features()

        # 2. 카드 사용 패턴 특징
        card_features = self._create_card_usage_features(meta_df)

        # 3. 리스크 프로필 특징
        risk_features = self._create_risk_profile_features(meta_df)

        # 4. 행동 패턴 특징
        behavioral_features = self._create_behavioral_features(meta_df)

        # 5. 교차 특징 (고객 × 카드 × 거래)
        cross_features = self._create_cross_features(meta_df)

        # 모든 특징 통합
        integrated_df = self._merge_all_features(
            meta_df, customer_features, card_features,
            risk_features, behavioral_features, cross_features
        )

        return integrated_df

    def _create_customer_profile_features(self):
        """고객 프로필 기반 특징"""
        features = self.users_df.copy()

        # 데이터 타입 전처리
        for col in ['yearly_income', 'total_debt', 'per_capita_income']:
            if col in features.columns:
                features[col] = (
                    features[col]
                    .astype(str)
                    .str.replace('[$,]', '', regex=True)
                    .str.replace('N/A', '0')
                    .replace('', '0')
                    .fillna('0')
                    .astype(float)
                )

        # 나이 그룹 세분화
        features['age_group_num'] = pd.cut(
            features['current_age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # 소득 대비 부채 비율
        features['debt_to_income_ratio'] = (
                features['total_debt'] / (features['yearly_income'] + 1)
        )

        # 신용 점수 등급
        features['credit_grade_num'] = pd.cut(
            features['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # 은퇴까지 남은 기간
        features['years_to_retirement'] = (
                features['retirement_age'] - features['current_age']
        )

        # 경제 활동 지수
        features['economic_activity_index'] = (
                features['yearly_income'] / features['per_capita_income'].replace(0, 1)
        )

        # 지리적 리스크 점수 (도시별 사기율 기반)
        features['geo_risk_score'] = self._calculate_geo_risk(features)

        return features

    def _create_card_usage_features(self, meta_df):
        """카드별 사용 패턴 특징"""

        # 카드 데이터 복사 및 전처리
        cards_df_processed = self.cards_df.copy()

        # credit_limit 데이터 타입 변환 (문자열 -> 숫자)
        if 'credit_limit' in cards_df_processed.columns:
            cards_df_processed['credit_limit'] = (
                cards_df_processed['credit_limit']
                .astype(str)
                .str.replace('[$,]', '', regex=True)
                .str.replace('N/A', '0')
                .replace('', '0')
                .fillna('0')
                .astype(float)
            )

        # 카드 정보와 거래 데이터 결합
        card_usage = meta_df.merge(
            cards_df_processed[['card_id', 'credit_limit', 'card_brand',
                                'card_type', 'has_chip', 'card_on_dark_web']],
            on='card_id',
            how='left'
        )

        # 카드별 집계 특징
        card_features = card_usage.groupby('card_id').agg({
            'last_amount': ['mean', 'std', 'max', 'sum'],
            'num_transactions_in_window': 'sum',
            'last_fraud': ['sum', 'mean'],
            'last_merchant_city': 'nunique',
            'last_mcc': 'nunique'
        }).reset_index()

        card_features.columns = ['card_id'] + [
            f'card_{c[0]}_{c[1]}' for c in card_features.columns[1:]
        ]

        # 신용 한도 대비 사용률
        card_features = card_features.merge(
            cards_df_processed[['card_id', 'credit_limit']], on='card_id'
        )

        card_features['utilization_rate'] = (
                card_features['card_last_amount_sum'] /
                (card_features['credit_limit'] + 1)
        )

        # 다크웹 노출 리스크
        card_features = card_features.merge(
            cards_df_processed[['card_id', 'card_on_dark_web']], on='card_id', how='left'
        )
        card_features['dark_web_risk'] = (
            card_features['card_on_dark_web']
            .map({'Yes': 1, 'No': 0}).fillna(0)
        )

        return card_features

    def _create_risk_profile_features(self, meta_df):
        """통합 리스크 프로필"""

        # 고객별 리스크 점수 계산
        customer_risk = self.users_df.copy()

        # 데이터 타입 변환
        for col in ['yearly_income', 'total_debt']:
            if col in customer_risk.columns:
                customer_risk[col] = (
                    customer_risk[col]
                    .astype(str)
                    .str.replace('[$,]', '', regex=True)
                    .str.replace('N/A', '0')
                    .replace('', '0')
                    .fillna('0')
                    .astype(float)
                )

        # 기본 리스크 요소
        customer_risk['base_risk_score'] = (
                (850 - customer_risk['credit_score']) / 850 * 0.3 +
                (customer_risk['total_debt'] / (customer_risk['yearly_income'] + 1)).clip(0, 1) * 0.3 +
                (customer_risk['current_age'] / 100) * 0.2 +
                (customer_risk['num_credit_cards'] / 10).clip(0, 1) * 0.2
        )

        # 카드별 리스크 집계
        cards_df_processed = self.cards_df.copy()

        # credit_limit 데이터 타입 변환
        if 'credit_limit' in cards_df_processed.columns:
            cards_df_processed['credit_limit'] = (
                cards_df_processed['credit_limit']
                .astype(str)
                .str.replace('[$,]', '', regex=True)
                .str.replace('N/A', '0')
                .replace('', '0')
                .fillna('0')
                .astype(float)
            )

        card_risk = cards_df_processed.groupby('client_id').agg({
            'card_on_dark_web': lambda x: (x == 'Yes').sum(),
            'credit_limit': 'sum',
            'card_id': 'count'
        }).reset_index()

        # 통합 리스크 프로필
        risk_features = customer_risk.merge(card_risk, on='client_id', how='left')

        # 복합 리스크 지표
        risk_features['composite_risk'] = (
                risk_features['base_risk_score'] * 0.5 +
                (risk_features['card_on_dark_web'] > 0).astype(float) * 0.3 +
                (risk_features['card_id'] > 5).astype(float) * 0.2
        )

        return risk_features

    def _create_behavioral_features(self, meta_df):
        """행동 패턴 기반 특징"""

        behavioral = meta_df.groupby('client_id').agg({
            'hour': lambda x: x.value_counts().index[0] if len(x) > 0 else 0,
            'day_of_week': lambda x: x.value_counts().index[0] if len(x) > 0 else 0,
            'last_amount': ['mean', 'std'],
            'num_transactions_in_window': ['mean', 'std'],
            'last_merchant_city': 'nunique',
            'last_mcc': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
            'last_fraud': ['sum', 'mean']
        }).reset_index()

        behavioral.columns = ['client_id'] + [
            f'behavior_{c[0]}_{c[1]}' if isinstance(c, tuple) else f'behavior_{c}'
            for c in behavioral.columns[1:]
        ]

        # 행동 일관성 점수
        behavioral['behavior_consistency'] = 1 / (
                behavioral['behavior_last_amount_std'] /
                (behavioral['behavior_last_amount_mean'] + 1) + 1
        )

        # 활동 다양성 지수
        behavioral['activity_diversity'] = (
                                                   behavioral['behavior_last_merchant_city_nunique'] *
                                                   behavioral.get('behavior_last_mcc_<lambda>', 1)
                                           ) ** 0.5

        return behavioral

    def _create_cross_features(self, meta_df):
        """고객×카드×거래 교차 특징"""

        # 고객-카드 조합별 특징
        cross_features = meta_df.groupby(['client_id', 'card_id']).agg({
            'last_amount': ['mean', 'std', 'max'],
            'last_fraud': 'sum',
            'num_transactions_in_window': 'max'
        }).reset_index()

        cross_features.columns = ['client_id', 'card_id'] + [
            f'cross_{c[0]}_{c[1]}' for c in cross_features.columns[2:]
        ]

        # 고객별 주 사용 카드 식별
        main_card = meta_df.groupby(['client_id', 'card_id']).size().reset_index(name='usage_count')
        main_card['is_main_card'] = main_card.groupby('client_id')['usage_count'].transform(
            lambda x: x == x.max()
        ).astype(int)

        cross_features = cross_features.merge(
            main_card[['client_id', 'card_id', 'is_main_card']],
            on=['client_id', 'card_id']
        )

        return cross_features

    def _calculate_geo_risk(self, features):
        """지리적 위치 기반 리스크 계산"""
        if 'latitude' in features.columns and 'longitude' in features.columns:
            lat_risk = np.abs(features['latitude'] - features['latitude'].median()) / (
                        features['latitude'].std() + 1e-6)
            lon_risk = np.abs(features['longitude'] - features['longitude'].median()) / (
                        features['longitude'].std() + 1e-6)
            return (lat_risk + lon_risk) / 2
        else:
            return 0

    def _merge_all_features(self, meta_df, *feature_dfs):
        """모든 특징 데이터프레임 통합"""

        result = meta_df.copy()

        for df in feature_dfs:
            if df is None or df.empty:
                continue

            if 'client_id' in df.columns and 'card_id' in df.columns:
                merge_on = ['client_id', 'card_id']
            elif 'client_id' in df.columns:
                merge_on = 'client_id'
            elif 'card_id' in df.columns:
                merge_on = 'card_id'
            else:
                continue

            # 카테고리형 컬럼을 숫자형으로 변환
            for col in df.select_dtypes(include=['category', 'object']).columns:
                if col not in merge_on:  # merge key는 제외
                    if df[col].dtype == 'category':
                        # 카테고리를 코드로 변환
                        df[col] = df[col].cat.codes
                    elif col in ['age_group', 'credit_grade']:
                        # 특정 카테고리형 컬럼 인코딩
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))

            result = result.merge(df, on=merge_on, how='left', suffixes=('', '_y'))

        # 중복 컬럼 제거
        result = result.loc[:, ~result.columns.duplicated()]

        # 결측치 처리 (숫자형 컬럼만)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

        # 무한대 값 처리
        result = result.replace([np.inf, -np.inf], 0)

        return result

    def _create_advanced_features(self, enhanced_features):
        """고급 특징 생성"""

        advanced = enhanced_features.copy()

        # 1. 시간 기반 고급 특징 (window_idx가 없으면 날짜 기반으로 생성)
        if 'window_idx_within_cust_card' in advanced.columns:
            advanced['transaction_velocity'] = advanced.groupby('client_id')[
                'window_idx_within_cust_card'].diff().fillna(0)
        else:
            # 날짜 기반 속도 계산
            advanced['window_end_date'] = pd.to_datetime(advanced['window_end_date'])
            advanced = advanced.sort_values(['client_id', 'window_end_date'])
            advanced['time_diff'] = advanced.groupby('client_id')[
                                        'window_end_date'].diff().dt.total_seconds() / 3600  # hours
            advanced['transaction_velocity'] = 1 / (advanced['time_diff'] + 1)

        advanced['amount_acceleration'] = advanced.groupby('client_id')['last_amount'].diff().diff().fillna(0)

        # 2. 고객별 행동 일관성 점수
        customer_consistency = advanced.groupby('client_id').agg({
            'last_amount': 'std',
            'hour': lambda x: len(set(x)) / 24,
            'last_merchant_city': 'nunique'
        }).reset_index()

        customer_consistency.columns = ['client_id', 'amount_consistency', 'time_diversity', 'location_diversity']
        advanced = advanced.merge(customer_consistency, on='client_id', how='left')

        # 3. 카드별 위험 점수
        if 'card_on_dark_web' in advanced.columns:
            advanced['card_risk_score'] = advanced['card_on_dark_web'].map({'Yes': 1, 'No': 0}).fillna(0)
        else:
            advanced['card_risk_score'] = 0

        # 4. 복합 위험 지표
        if 'customer_risk_score' not in advanced.columns:
            advanced['customer_risk_score'] = 0
        if 'amount_vs_profile_ratio' not in advanced.columns:
            advanced['amount_vs_profile_ratio'] = 1

        advanced['composite_risk'] = (
                advanced['customer_risk_score'] * 0.4 +
                advanced['card_risk_score'] * 0.3 +
                (advanced['amount_vs_profile_ratio'] > 2).astype(int) * 0.3
        )

        # 5. 네트워크 특징
        merchant_fraud_rate = advanced.groupby('last_merchant_city')['last_fraud'].mean().to_dict()
        advanced['merchant_fraud_rate'] = advanced['last_merchant_city'].map(merchant_fraud_rate).fillna(0)

        # 추가 리스크 관련 컬럼들
        if 'risk_score' in advanced.columns:
            advanced['customer_risk_score'] = advanced['risk_score']
        if 'base_risk_score' in advanced.columns:
            advanced['base_risk_score'] = advanced['base_risk_score']

        return advanced