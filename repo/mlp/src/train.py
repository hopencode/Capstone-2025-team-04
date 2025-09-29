import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from .model import EnhancedCustomerAwareEnsemble
from .feature_engineering import CustomerCardFeatureEngineering
from .utils import FocalLoss, apply_smote_sampling # utils.py에서 import


class EnhancedFraudDetectionSystem:
    """고객/카드 정보와 앙상블을 결합한 고급 사기 탐지 시스템"""

    def __init__(self, users_df, cards_df):
        self.users_df = users_df
        self.cards_df = cards_df
        self.feature_engineer = CustomerCardFeatureEngineering(users_df, cards_df)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enhanced_models = {}
        self.feature_scalers = {}

    def create_enhanced_datasets(self, meta_df, embeddings_df):
        """기존 특징 + 고객/카드 특징을 결합한 데이터셋 생성"""

        print("Enhanced feature engineering 시작...")

        # 1. 기존 특징들 생성
        print("기존 특징 생성 중...")
        method1_X, method1_y = self._create_method1_enhanced(meta_df, embeddings_df)
        method2_X, method2_y = self._create_method2_enhanced(meta_df)
        method4_X, method4_y = self._create_method4_enhanced(meta_df)

        # 2. 고객/카드 통합 특징 생성
        print("고객/카드 특징 생성 중...")
        integrated_features_df = self.feature_engineer.create_integrated_features(meta_df)

        # 3. 고급 특징들 추가 생성
        print("고급 특징 생성 중...")
        advanced_features = self._create_advanced_features(integrated_features_df)

        # 4. 모든 특징 결합
        print("특징 결합 중...")
        labels = integrated_features_df['last_fraud'].values

        # 숫자형 컬럼만 선택하는 함수
        def get_numeric_columns(df, prefix_list):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return [col for col in numeric_cols if any(col.startswith(prefix) for prefix in prefix_list)]

        # 각 특징 그룹을 명시적으로 분리 (숫자형만)
        customer_prefixes = ['age_', 'credit_', 'debt_', 'economic_', 'years_', 'geo_',
                             'customer_', 'income_', 'behavior_']
        card_prefixes = ['card_', 'utilization_', 'dark_web_']
        risk_prefixes = ['risk_', 'composite_', 'base_']

        customer_cols = get_numeric_columns(advanced_features, customer_prefixes)
        card_cols = get_numeric_columns(advanced_features, card_prefixes)
        risk_cols = get_numeric_columns(advanced_features, risk_prefixes)

        # 빈 컬럼 리스트 처리
        if not customer_cols:
            # 기본 숫자형 컬럼 찾기
            potential_customer_cols = ['debt_to_income_ratio', 'economic_activity_index',
                                       'years_to_retirement', 'geo_risk_score']
            customer_cols = [col for col in potential_customer_cols if col in advanced_features.columns]
            if not customer_cols:
                advanced_features['customer_default'] = 0.0
                customer_cols = ['customer_default']

        if not card_cols:
            potential_card_cols = ['utilization_rate', 'dark_web_risk', 'card_risk_score']
            card_cols = [col for col in potential_card_cols if col in advanced_features.columns]
            if not card_cols:
                advanced_features['card_default'] = 0.0
                card_cols = ['card_default']

        if not risk_cols:
            potential_risk_cols = ['composite_risk', 'base_risk_score', 'merchant_fraud_rate']
            risk_cols = [col for col in potential_risk_cols if col in advanced_features.columns]
            if not risk_cols:
                advanced_features['risk_default'] = 0.0
                risk_cols = ['risk_default']

        # 데이터 추출 (숫자형만 선택하고 fillna 적용)
        X_customer = advanced_features[customer_cols].values.astype(np.float32)
        X_card = advanced_features[card_cols].values.astype(np.float32)
        X_risk = advanced_features[risk_cols].values.astype(np.float32)

        # NaN 처리
        X_customer = np.nan_to_num(X_customer, 0)
        X_card = np.nan_to_num(X_card, 0)
        X_risk = np.nan_to_num(X_risk, 0)

        combined_datasets = {
            'enhanced_embedding': {'X': method1_X, 'y': labels},
            'enhanced_temporal': {'X': method2_X, 'y': labels},
            'enhanced_anomaly': {'X': method4_X, 'y': labels},
            'enhanced_meta_features': {
                'X_customer': X_customer,
                'X_card': X_card,
                'X_risk': X_risk,
                'y': labels
            }
        }

        # 스케일러 학습 및 저장
        self.feature_scalers['customer'] = StandardScaler().fit(X_customer)
        self.feature_scalers['card'] = StandardScaler().fit(X_card)
        self.feature_scalers['risk'] = StandardScaler().fit(X_risk)

        return combined_datasets

    def _create_method1_enhanced(self, meta_df, embeddings_df):
        """Enhanced Method 1: 임베딩 + 메타 + 고객정보"""

        embedding_features = embeddings_df.iloc[:, 1:].values

        enriched_meta = meta_df.copy()
        enriched_meta['window_end_date'] = pd.to_datetime(enriched_meta['window_end_date'])
        enriched_meta['hour'] = enriched_meta['window_end_date'].dt.hour
        enriched_meta['day_of_week'] = enriched_meta['window_end_date'].dt.dayofweek
        enriched_meta['month'] = enriched_meta['window_end_date'].dt.month
        enriched_meta['is_weekend'] = enriched_meta['day_of_week'].isin([5, 6]).astype(int)
        enriched_meta['is_night'] = enriched_meta['hour'].between(22, 6).astype(int)
        enriched_meta['is_business_hour'] = enriched_meta['hour'].between(9, 17).astype(int)

        # 고객 정보 병합
        customer_info = self.users_df[['client_id', 'current_age', 'credit_score',
                                       'yearly_income', 'total_debt', 'num_credit_cards']].copy()

        for col in ['yearly_income', 'total_debt']:
            if col in customer_info.columns:
                customer_info[col] = (
                    customer_info[col]
                    .astype(str)
                    .str.replace('[$,]', '', regex=True)
                    .str.replace('N/A', '0')
                    .replace('', '0')
                    .fillna('0')
                    .astype(float)
                )

        # 정규화된 고객 특징
        customer_info['age_normalized'] = customer_info['current_age'] / 100
        customer_info['credit_normalized'] = customer_info['credit_score'] / 850
        customer_info['income_log'] = np.log1p(customer_info['yearly_income'])
        customer_info['debt_ratio'] = customer_info['total_debt'] / (customer_info['yearly_income'] + 1)

        enriched_meta = enriched_meta.merge(customer_info, on='client_id', how='left')

        # 수치형 특징 선택 (window_idx_within_cust_card 제거)
        numeric_features = [
            'last_amount', 'num_transactions_in_window',
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'is_business_hour',
            'age_normalized', 'credit_normalized', 'income_log', 'debt_ratio', 'num_credit_cards'
        ]

        meta_features = enriched_meta[numeric_features].fillna(0).values
        combined_features = np.concatenate([embedding_features, meta_features], axis=1)

        return combined_features, enriched_meta['last_fraud'].values

    def _create_method2_enhanced(self, meta_df):
        """Enhanced Method 2: 시계열 + 고객 컨텍스트"""

        df = meta_df.copy()
        df['window_end_date'] = pd.to_datetime(df['window_end_date'])
        df['hour'] = df['window_end_date'].dt.hour
        df['day_of_week'] = df['window_end_date'].dt.dayofweek
        df['month'] = df['window_end_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # 고객 정보 추가
        customer_context = self.users_df[['client_id', 'current_age', 'credit_score', 'yearly_income']].copy()

        if 'yearly_income' in customer_context.columns:
            customer_context['yearly_income'] = (
                customer_context['yearly_income']
                .astype(str)
                .str.replace('[$,]', '', regex=True)
                .str.replace('N/A', '0')
                .replace('', '0')
                .fillna('0')
                .astype(float)
            )

        df = df.merge(customer_context, on='client_id', how='left')

        # 수치형 변환
        numeric_cols = ['last_amount', 'last_mcc', 'last_zip', 'current_age', 'credit_score', 'yearly_income']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 범주형 인코딩
        le_city = LabelEncoder()
        le_state = LabelEncoder()
        le_chip = LabelEncoder()

        df['last_merchant_city'] = df['last_merchant_city'].astype(str).fillna('UNKNOWN')
        df['last_merchant_state'] = df['last_merchant_state'].astype(str).fillna('UNKNOWN')
        df['last_use_chip'] = df['last_use_chip'].astype(str).fillna('UNKNOWN')

        df['city_encoded'] = le_city.fit_transform(df['last_merchant_city'])
        df['state_encoded'] = le_state.fit_transform(df['last_merchant_state'])
        df['chip_encoded'] = le_chip.fit_transform(df['last_use_chip'])

        # window_idx 생성 (날짜 기반)
        df = df.sort_values(['client_id', 'card_id', 'window_end_date'])
        df['window_idx'] = df.groupby(['client_id', 'card_id']).cumcount()

        # 특징 목록 (window_idx_within_cust_card 대신 window_idx 사용)
        feature_cols = [
            'last_amount', 'num_transactions_in_window', 'window_idx',
            'last_zip', 'last_mcc', 'hour', 'day_of_week', 'month',
            'is_weekend', 'city_encoded', 'state_encoded', 'chip_encoded',
            'current_age', 'credit_score', 'yearly_income'
        ]

        # 존재하는 컬럼만 선택
        feature_cols = [col for col in feature_cols if col in df.columns]

        sequence_length = 5
        X_list = []
        y_list = []

        groups = df.groupby(['client_id', 'card_id'])

        for (client_id, card_id), group in tqdm(groups, desc='Enhanced Temporal 특징 생성'):
            group = group.sort_values('window_end_date').reset_index(drop=True)

            if len(group) < sequence_length:
                continue

            for i in range(len(group) - sequence_length + 1):
                sequence_data = group.iloc[i:i + sequence_length]

                sequence_features = []
                for _, row in sequence_data.iterrows():
                    window_features = [float(row[col]) if not pd.isna(row[col]) else 0.0
                                       for col in feature_cols]
                    sequence_features.extend(window_features)

                label = sequence_data.iloc[-1]['last_fraud']

                X_list.append(sequence_features)
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32) if X_list else np.array([]).reshape(0,
                                                                                   len(feature_cols) * sequence_length)
        y = np.array(y_list) if y_list else np.array([])

        return X, y

    def _create_method4_enhanced(self, meta_df):
        """Enhanced Method 4: 이상 패턴 + 고객 프로필"""

        df = meta_df.copy()
        df['window_end_date'] = pd.to_datetime(df['window_end_date'])
        df['hour'] = df['window_end_date'].dt.hour
        df['day_of_week'] = df['window_end_date'].dt.dayofweek
        df['month'] = df['window_end_date'].dt.month

        # 고객 리스크 프로필 추가
        customer_risk = self.users_df[['client_id', 'current_age', 'credit_score',
                                       'yearly_income', 'total_debt']].copy()

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

        customer_risk['risk_score'] = (850 - customer_risk['credit_score']) / 850
        customer_risk['debt_ratio'] = customer_risk['total_debt'] / (customer_risk['yearly_income'] + 1)

        df = df.merge(customer_risk, on='client_id', how='left')

        numeric_cols = ['last_amount', 'last_mcc', 'current_age', 'credit_score',
                        'yearly_income', 'total_debt', 'risk_score', 'debt_ratio']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 범주형 인코딩
        le_city = LabelEncoder()
        le_state = LabelEncoder()
        le_chip = LabelEncoder()

        df['last_merchant_city'] = df['last_merchant_city'].astype(str).fillna('UNKNOWN')
        df['last_merchant_state'] = df['last_merchant_state'].astype(str).fillna('UNKNOWN')
        df['last_use_chip'] = df['last_use_chip'].astype(str).fillna('UNKNOWN')

        df['city_encoded'] = le_city.fit_transform(df['last_merchant_city'])
        df['state_encoded'] = le_state.fit_transform(df['last_merchant_state'])
        df['chip_encoded'] = le_chip.fit_transform(df['last_use_chip'])

        features_list = []
        groups = df.groupby(['client_id', 'card_id'])

        for (client_id, card_id), group in tqdm(groups, desc='Enhanced Anomaly 특징 생성'):
            group = group.sort_values('window_end_date').reset_index(drop=True)

            # 개인별/카드별 통계
            personal_stats = {
                'amount_mean': group['last_amount'].mean(),
                'amount_std': group['last_amount'].std() if len(group) > 1 else 1.0,
                'amount_q25': group['last_amount'].quantile(0.25),
                'amount_q75': group['last_amount'].quantile(0.75),
            }

            for idx, current_row in group.iterrows():
                features = {}
                history = group.iloc[:idx + 1]
                current_amount = current_row['last_amount']

                # 기본 이상 탐지 특징들
                features['amount_zscore_global'] = abs(
                    (current_amount - personal_stats['amount_mean']) / (personal_stats['amount_std'] + 1e-6))
                iqr = personal_stats['amount_q75'] - personal_stats['amount_q25'] + 1e-6
                features['amount_outlier_iqr'] = int(
                    current_amount < (personal_stats['amount_q25'] - 1.5 * iqr) or
                    current_amount > (personal_stats['amount_q75'] + 1.5 * iqr)
                )

                # 고객 리스크 기반 특징들
                features['customer_risk_score'] = current_row.get('risk_score', 0)
                features['customer_debt_ratio'] = current_row.get('debt_ratio', 0)
                features['customer_age_normalized'] = current_row.get('current_age', 0) / 100
                features['customer_credit_normalized'] = current_row.get('credit_score', 700) / 850

                # 금액 대비 고객 프로필 이상도
                expected_amount = current_row.get('yearly_income', 50000) / 365 * 0.1
                features['amount_vs_profile_ratio'] = current_amount / (expected_amount + 1)

                # 신용도 대비 거래 이상도
                credit_factor = current_row.get('credit_score', 700) / 850
                features['amount_vs_credit_ratio'] = current_amount * (1 - credit_factor)

                # 시간 패턴 특징들
                features['current_hour'] = current_row['hour']
                features['current_day'] = current_row['day_of_week']
                features['current_month'] = current_row['month']
                features['is_unusual_hour'] = int(current_row['hour'] < 6 or current_row['hour'] > 22)

                # 거래 빈도 특징들
                features['transaction_count_history'] = len(history)
                features['amount_percentile'] = (history['last_amount'] < current_amount).mean()

                # 상점 패턴 특징들
                mcc_frequency = (history['last_mcc'] == current_row['last_mcc']).sum()
                features['mcc_frequency'] = mcc_frequency / max(len(history), 1)
                features['is_new_mcc'] = int(mcc_frequency == 1)

                city_frequency = (history['city_encoded'] == current_row['city_encoded']).sum()
                features['city_frequency'] = city_frequency / max(len(history), 1)
                features['is_new_city'] = int(city_frequency == 1)

                features['fraud'] = current_row['last_fraud']
                features_list.append(features)

        if not features_list:
            # 빈 데이터셋 처리
            return np.array([]).reshape(0, 20), np.array([])

        features_df = pd.DataFrame(features_list).fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # 특징 선택 (고객 정보 포함 확장)
        feature_cols = [col for col in features_df.columns if col != 'fraud']

        X = features_df[feature_cols].values
        y = features_df['fraud'].values

        return X, y

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

def train_epoch(model, X_customer, X_card, X_risk, X_ensemble, y, optimizer, criterion):
    """한 에폭 학습 함수"""
    model.train()
    optimizer.zero_grad()
    outputs = model(X_customer, X_card, X_risk, X_ensemble)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, X_customer, X_card, X_risk, X_ensemble, y, criterion):
    """검증 함수"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_customer, X_card, X_risk, X_ensemble)
        loss = criterion(outputs, y)
        predictions = outputs.cpu().numpy().flatten()
        auc = roc_auc_score(y.cpu().numpy(), predictions)
    return loss.item(), auc, predictions

def run_enhanced_fraud_detection_system(
        train_meta_df, train_embeddings_df,
        val_meta_df, val_embeddings_df,
        test_meta_df, test_embeddings_df,
        users_df, cards_df,
        use_focal_loss=True,
        use_smote=False,
        use_optimal_threshold=True
):
    """개선된 Enhanced Fraud Detection System"""

    print("Enhanced Fraud Detection System 시작!")
    print("=" * 80)

    # 1. 데이터 전처리 통합
    print("데이터 전처리 중...")

    if 'id' in users_df.columns:
        users_df = users_df.rename(columns={'id': 'client_id'})
    if 'id' in cards_df.columns:
        cards_df = cards_df.rename(columns={'id': 'card_id'})

    # 2. 시스템 초기화 및 각 데이터셋 생성
    enhanced_system = EnhancedFraudDetectionSystem(users_df, cards_df)

    print("\nTrain 데이터셋 생성 중...")
    train_datasets = enhanced_system.create_enhanced_datasets(train_meta_df, train_embeddings_df)

    print("\nValidation 데이터셋 생성 중...")
    val_datasets = enhanced_system.create_enhanced_datasets(val_meta_df, val_embeddings_df)

    print("\nTest 데이터셋 생성 중...")
    test_datasets = enhanced_system.create_enhanced_datasets(test_meta_df, test_embeddings_df)

    print(f"\n생성된 데이터셋 크기:")
    print(f"  Train: {len(train_datasets['enhanced_meta_features']['y'])}")
    print(f"  Validation: {len(val_datasets['enhanced_meta_features']['y'])}")
    print(f"  Test: {len(test_datasets['enhanced_meta_features']['y'])}")

    # 3. 데이터 추출 및 처리
    print("\n데이터 처리 중...")

    train_min_size = min(
        len(train_datasets['enhanced_meta_features']['y']),
        len(train_datasets.get('enhanced_embedding', {'X': []})['X']),
        len(train_datasets.get('enhanced_temporal', {'X': []})['X']),
        len(train_datasets.get('enhanced_anomaly', {'X': []})['X'])
    )

    val_min_size = min(
        len(val_datasets['enhanced_meta_features']['y']),
        len(val_datasets.get('enhanced_embedding', {'X': []})['X']),
        len(val_datasets.get('enhanced_temporal', {'X': []})['X']),
        len(val_datasets.get('enhanced_anomaly', {'X': []})['X'])
    )

    test_min_size = min(
        len(test_datasets['enhanced_meta_features']['y']),
        len(test_datasets.get('enhanced_embedding', {'X': []})['X']),
        len(test_datasets.get('enhanced_temporal', {'X': []})['X']),
        len(test_datasets.get('enhanced_anomaly', {'X': []})['X'])
    )

    # 모든 데이터셋을 최소 크기로 조정
    for name in ['embedding', 'temporal', 'anomaly']:
        dataset_name = f'enhanced_{name}'
        if dataset_name in train_datasets:
            train_datasets[dataset_name]['X'] = train_datasets[dataset_name]['X'][:train_min_size]
        if dataset_name in val_datasets:
            val_datasets[dataset_name]['X'] = val_datasets[dataset_name]['X'][:val_min_size]
        if dataset_name in test_datasets:
            test_datasets[dataset_name]['X'] = test_datasets[dataset_name]['X'][:test_min_size]

    train_datasets['enhanced_meta_features']['X_customer'] = train_datasets['enhanced_meta_features']['X_customer'][
                                                             :train_min_size]
    train_datasets['enhanced_meta_features']['X_card'] = train_datasets['enhanced_meta_features']['X_card'][
                                                         :train_min_size]
    train_datasets['enhanced_meta_features']['X_risk'] = train_datasets['enhanced_meta_features']['X_risk'][
                                                         :train_min_size]
    train_datasets['enhanced_meta_features']['y'] = train_datasets['enhanced_meta_features']['y'][:train_min_size]

    val_datasets['enhanced_meta_features']['X_customer'] = val_datasets['enhanced_meta_features']['X_customer'][
                                                           :val_min_size]
    val_datasets['enhanced_meta_features']['X_card'] = val_datasets['enhanced_meta_features']['X_card'][:val_min_size]
    val_datasets['enhanced_meta_features']['X_risk'] = val_datasets['enhanced_meta_features']['X_risk'][:val_min_size]
    val_datasets['enhanced_meta_features']['y'] = val_datasets['enhanced_meta_features']['y'][:val_min_size]

    test_datasets['enhanced_meta_features']['X_customer'] = test_datasets['enhanced_meta_features']['X_customer'][
                                                            :test_min_size]
    test_datasets['enhanced_meta_features']['X_card'] = test_datasets['enhanced_meta_features']['X_card'][
                                                        :test_min_size]
    test_datasets['enhanced_meta_features']['X_risk'] = test_datasets['enhanced_meta_features']['X_risk'][
                                                        :test_min_size]
    test_datasets['enhanced_meta_features']['y'] = test_datasets['enhanced_meta_features']['y'][:test_min_size]

    y_train_all = train_datasets['enhanced_meta_features']['y']
    y_val_all = val_datasets['enhanced_meta_features']['y']
    y_test_all = test_datasets['enhanced_meta_features']['y']

    train_valid_mask = ~np.isnan(y_train_all)
    val_valid_mask = ~np.isnan(y_val_all)
    test_valid_mask = ~np.isnan(y_test_all)

    y_train = y_train_all[train_valid_mask]
    y_val = y_val_all[val_valid_mask]
    y_test = y_test_all[test_valid_mask]

    print(f"최종 데이터 크기:")
    print(f"  Train: {len(y_train)} (Fraud: {y_train.mean():.3f})")
    print(f"  Val: {len(y_val)} (Fraud: {y_val.mean():.3f})")
    print(f"  Test: {len(y_test)} (Fraud: {y_test.mean():.3f})")

    # 4. 개선된 앙상블 예측값 생성 (확률 분포 전체 사용)
    print("\n기존 모델 예측값 생성 중...")

    # 6차원 앙상블 (3 models × 2 classes)
    X_ensemble_train = np.zeros((len(y_train), 6))
    X_ensemble_val = np.zeros((len(y_val), 6))
    X_ensemble_test = np.zeros((len(y_test), 6))

    for idx, name in enumerate(['embedding', 'temporal', 'anomaly']):
        dataset_name = f'enhanced_{name}'
        if dataset_name in train_datasets:
            X_data_train = train_datasets[dataset_name]['X'][train_valid_mask]
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X_data_train, y_train)

            # 전체 확률 분포 저장
            X_ensemble_train[:, idx * 2:(idx + 1) * 2] = rf.predict_proba(X_data_train)

            if dataset_name in val_datasets:
                X_data_val = val_datasets[dataset_name]['X'][val_valid_mask]
                X_ensemble_val[:, idx * 2:(idx + 1) * 2] = rf.predict_proba(X_data_val)

            if dataset_name in test_datasets:
                X_data_test = test_datasets[dataset_name]['X'][test_valid_mask]
                X_ensemble_test[:, idx * 2:(idx + 1) * 2] = rf.predict_proba(X_data_test)

    # 5. Enhanced 모델 학습 준비
    print("\nEnhanced 모델 학습 준비 중...")

    X_customer_train = train_datasets['enhanced_meta_features']['X_customer'][train_valid_mask]
    X_card_train = train_datasets['enhanced_meta_features']['X_card'][train_valid_mask]
    X_risk_train = train_datasets['enhanced_meta_features']['X_risk'][train_valid_mask]

    X_customer_val = val_datasets['enhanced_meta_features']['X_customer'][val_valid_mask]
    X_card_val = val_datasets['enhanced_meta_features']['X_card'][val_valid_mask]
    X_risk_val = val_datasets['enhanced_meta_features']['X_risk'][val_valid_mask]

    X_customer_test = test_datasets['enhanced_meta_features']['X_customer'][test_valid_mask]
    X_card_test = test_datasets['enhanced_meta_features']['X_card'][test_valid_mask]
    X_risk_test = test_datasets['enhanced_meta_features']['X_risk'][test_valid_mask]

    # SMOTE 적용 (옵션)
    if use_smote:
        print("\nSMOTE 리샘플링 적용 중...")
        X_all_train = np.concatenate([X_customer_train, X_card_train, X_risk_train, X_ensemble_train], axis=1)

        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X_all_resampled, y_train_resampled = smote.fit_resample(X_all_train, y_train)

        print(f"리샘플링 전: {len(y_train)} → 리샘플링 후: {len(y_train_resampled)}")
        print(f"Fraud 비율: {y_train.mean():.3f} → {y_train_resampled.mean():.3f}")

        # 리샘플링된 데이터 분리
        idx = 0
        customer_dim = X_customer_train.shape[1]
        card_dim = X_card_train.shape[1]
        risk_dim = X_risk_train.shape[1]

        X_customer_train = X_all_resampled[:, idx:idx + customer_dim]
        idx += customer_dim
        X_card_train = X_all_resampled[:, idx:idx + card_dim]
        idx += card_dim
        X_risk_train = X_all_resampled[:, idx:idx + risk_dim]
        idx += risk_dim
        X_ensemble_train = X_all_resampled[:, idx:]
        y_train = y_train_resampled

    # 스케일링
    X_customer_train = enhanced_system.feature_scalers['customer'].fit_transform(X_customer_train)
    X_card_train = enhanced_system.feature_scalers['card'].fit_transform(X_card_train)
    X_risk_train = enhanced_system.feature_scalers['risk'].fit_transform(X_risk_train)

    X_customer_val = enhanced_system.feature_scalers['customer'].transform(X_customer_val)
    X_card_val = enhanced_system.feature_scalers['card'].transform(X_card_val)
    X_risk_val = enhanced_system.feature_scalers['risk'].transform(X_risk_val)

    X_customer_test = enhanced_system.feature_scalers['customer'].transform(X_customer_test)
    X_card_test = enhanced_system.feature_scalers['card'].transform(X_card_test)
    X_risk_test = enhanced_system.feature_scalers['risk'].transform(X_risk_test)

    # 텐서 변환
    device = enhanced_system.device

    X_customer_train_tensor = torch.FloatTensor(X_customer_train).to(device)
    X_card_train_tensor = torch.FloatTensor(X_card_train).to(device)
    X_risk_train_tensor = torch.FloatTensor(X_risk_train).to(device)
    X_ensemble_train_tensor = torch.FloatTensor(X_ensemble_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    X_customer_val_tensor = torch.FloatTensor(X_customer_val).to(device)
    X_card_val_tensor = torch.FloatTensor(X_card_val).to(device)
    X_risk_val_tensor = torch.FloatTensor(X_risk_val).to(device)
    X_ensemble_val_tensor = torch.FloatTensor(X_ensemble_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    X_customer_test_tensor = torch.FloatTensor(X_customer_test).to(device)
    X_card_test_tensor = torch.FloatTensor(X_card_test).to(device)
    X_risk_test_tensor = torch.FloatTensor(X_risk_test).to(device)
    X_ensemble_test_tensor = torch.FloatTensor(X_ensemble_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # 개선된 모델 초기화
    enhanced_model = EnhancedCustomerAwareEnsemble(
        customer_dim=X_customer_train.shape[1],  # 22
        card_dim=X_card_train.shape[1],  # 14
        risk_dim=X_risk_train.shape[1],  # 2
        ensemble_dim=6,  # 6차원 (3 models × 2 classes)
        hidden_dim=256
    ).to(device)

    # Loss Function 설정
    if use_focal_loss:
        print("Focal Loss 사용")
        criterion = FocalLoss(alpha=0.25, gamma=2)
    else:
        print("BCEWithLogitsLoss 사용 (클래스 가중치 적용)")
        pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
        print(f"Positive weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(enhanced_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 학습 루프
    print("\nEnhanced 모델 학습 중...")
    epochs = 100
    best_val_auc = 0
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        # 학습
        enhanced_model.train()
        optimizer.zero_grad()

        # Logits 출력
        logits = enhanced_model(
            X_customer_train_tensor,
            X_card_train_tensor,
            X_risk_train_tensor,
            X_ensemble_train_tensor
        )

        train_loss = criterion(logits, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # 검증
        enhanced_model.eval()
        with torch.no_grad():
            val_logits = enhanced_model(
                X_customer_val_tensor,
                X_card_val_tensor,
                X_risk_val_tensor,
                X_ensemble_val_tensor
            )
            val_loss = criterion(val_logits, y_val_tensor)

            # Sigmoid 적용하여 확률로 변환
            val_preds = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_preds)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(enhanced_model.state_dict(), 'best_enhanced_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss.item():.4f}, Val AUC={val_auc:.4f}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\n최고 검증 AUC: {best_val_auc:.4f}")

    # 6. 최종 테스트 평가
    print("\n최종 테스트 평가...")
    enhanced_model.load_state_dict(torch.load('best_enhanced_model.pth'))
    enhanced_model.eval()

    with torch.no_grad():
        test_logits = enhanced_model(
            X_customer_test_tensor,
            X_card_test_tensor,
            X_risk_test_tensor,
            X_ensemble_test_tensor
        )
        test_preds = torch.sigmoid(test_logits).cpu().numpy().flatten()

    # 최적 임계값 찾기
    if use_optimal_threshold:
        print("\n최적 임계값 탐색 중...")
        precision, recall, thresholds = precision_recall_curve(y_test, test_preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]

        print(f"최적 임계값: {optimal_threshold:.3f} (기본: 0.5)")
        print(f"최적 F1 Score: {f1_scores[optimal_idx]:.4f}")

        test_binary = (test_preds > optimal_threshold).astype(int)
    else:
        optimal_threshold = 0.5
        test_binary = (test_preds > 0.5).astype(int)

    test_auc = roc_auc_score(y_test, test_preds)
    test_acc = accuracy_score(y_test, test_binary)

    # 7. 추가 ML 모델들 학습
    print("\n추가 ML 모델 학습 중...")
    ml_results = {}

    X_all_train = np.concatenate([X_customer_train, X_card_train, X_risk_train, X_ensemble_train], axis=1)
    X_all_test = np.concatenate([X_customer_test, X_card_test, X_risk_test, X_ensemble_test], axis=1)

    # 클래스 가중치 계산
    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', random_state=42, max_iter=1000
        )
    }

    for model_name, model in models.items():
        model.fit(X_all_train, y_train)
        pred = model.predict_proba(X_all_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        acc = accuracy_score(y_test, pred > optimal_threshold)
        ml_results[model_name] = {'auc': auc, 'acc': acc, 'predictions': pred}
        print(f"  {model_name}: AUC={auc:.4f}, Accuracy={acc:.4f}")

    # 8. 최종 앙상블 (가중 평균)
    weights = [0.4, 0.25, 0.25, 0.1]  # Enhanced, RF, XGB, LR
    all_predictions = [test_preds] + [result['predictions'] for result in ml_results.values()]
    final_ensemble_pred = np.average(all_predictions, weights=weights, axis=0)

    final_auc = roc_auc_score(y_test, final_ensemble_pred)
    final_acc = accuracy_score(y_test, final_ensemble_pred > optimal_threshold)

    print("\n" + "=" * 80)
    print("최종 성능 (Enhanced Fraud Detection System)")
    print("=" * 80)
    print(f"Enhanced Neural Network: AUC={test_auc:.4f}, Accuracy={test_acc:.4f}")
    print(f"Final Ensemble: AUC={final_auc:.4f}, Accuracy={final_acc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print("\n상세 분류 보고서:")
    print(classification_report(y_test, final_ensemble_pred > optimal_threshold, target_names=['Normal', 'Fraud']))

    # 혼동 행렬
    cm = confusion_matrix(y_test, final_ensemble_pred > optimal_threshold)
    print("\n혼동 행렬:")
    print(f"    True Negative:  {cm[0, 0]:6d}    False Positive: {cm[0, 1]:6d}")
    print(f"    False Negative: {cm[1, 0]:6d}    True Positive:  {cm[1, 1]:6d}")

    # 결과 저장
    results = {
        'enhanced_model': enhanced_model,
        'test_auc': test_auc,
        'final_auc': final_auc,
        'ml_results': ml_results,
        'predictions': {
            'enhanced': test_preds,
            'ensemble': final_ensemble_pred
        },
        'y_test': y_test,
        'optimal_threshold': optimal_threshold
    }

    print("\nEnhanced Fraud Detection System 완료!")
    return enhanced_model, results