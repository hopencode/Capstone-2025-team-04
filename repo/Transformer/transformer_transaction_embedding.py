import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
from tqdm import tqdm
import os
from datetime import timedelta
import pickle

# 동일 입력 동일 결과를 위한 랜덤 시드 고정
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 1. 데이터 전처리 및 피처 계산
def preprocess_and_feature_engineer(df, le_mcc=None, le_zip=None, le_region=None, le_use_chip=None, scalers=None):
    df['date'] = pd.to_datetime(df['date'])
    df['transaction_id'] = np.arange(len(df))
    df = df.sort_values(['client_id', 'card_id', 'date', 'transaction_id'])
    df['is_online'] = df['is_online'].fillna((df['merchant_region'] == 'ONLINE').astype(int)).astype(int)
    df['prev_date'] = df.groupby(['client_id', 'card_id'])['date'].shift(1)
    df['delta_time'] = (df['date'] - df['prev_date']).dt.total_seconds()
    df['prev_is_online'] = df.groupby(['client_id', 'card_id'])['is_online'].shift(1)

    def haversine(lat1, lon1, lat2, lon2):
        cond = np.isnan(lat1) | np.isnan(lon1) | np.isnan(lat2) | np.isnan(lon2)
        lat1, lon1, lat2, lon2 = (
            np.where(cond, 0, lat1), np.where(cond, 0, lon1), np.where(cond, 0, lat2), np.where(cond, 0, lon2))
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = phi2 - phi1
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    df['prev_lat'] = df.groupby(['client_id', 'card_id'])['latitude'].shift(1)
    df['prev_lon'] = df.groupby(['client_id', 'card_id'])['longitude'].shift(1)
    df['delta_distance'] = np.where(
        (df['is_online'] == 0) & (df['prev_is_online'] == 0),
        haversine(df['prev_lat'], df['prev_lon'], df['latitude'], df['longitude']),
        np.nan)
    df['is_delta_time_missing'] = df['delta_time'].isna().astype(int)
    df['is_delta_distance_missing'] = df['delta_distance'].isna().astype(int)
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

    card_features = pd.read_csv("./data/cleaned_cards.csv")

    df = df.merge(card_features[['id', 'acct_open_date']], left_on='card_id', right_on='id', how='left',
                  suffixes=('', '_acct'))
    df['acct_open_date'] = pd.to_datetime(df['acct_open_date'])
    df['account_age_days'] = (df['date'] - df['acct_open_date']).dt.days.fillna(0)
    df.drop(columns=['id_acct'], inplace=True, errors='ignore')
    df = df.merge(card_features[['id', 'expires']], left_on='card_id', right_on='id', how='left', suffixes=('', '_exp'))
    df['expires_date'] = pd.to_datetime(df['expires'], format='%m/%Y', errors='coerce')
    df['months_to_expiry'] = ((df['expires_date'] - df['date']).dt.days // 30).fillna(0)
    df.drop(columns=['id_exp'], inplace=True, errors='ignore')
    df['mcc'] = df['mcc'].astype(str).fillna("NaN")
    df['zip'] = df['zip'].astype(str).fillna("NaN")
    df['merchant_region'] = df['merchant_region'].fillna("NaN")
    df['use_chip'] = df['use_chip'].fillna("NaN")

    def fit_or_transform_label(le, series, fit=True):
        if fit:
            le = LabelEncoder().fit(series)
            le.classes_ = np.insert(le.classes_, 0, 'UNK')
        known_labels = set(le.classes_)
        transformed_series = series.apply(lambda x: x if x in known_labels else 'UNK')
        return le.transform(transformed_series) + 1, le

    def fit_or_transform_scaler(scaler, df_col, fit=True):
        nan_mask = df_col.isna()
        if fit:
            scaler = MinMaxScaler()
            df_col_non_nan = df_col[~nan_mask].values.reshape(-1, 1)
            if df_col_non_nan.size > 0:
                scaler.fit(df_col_non_nan)
            else:
                return np.zeros_like(df_col.values.reshape(-1, 1)), scaler
        df_col_temp = df_col.fillna(0.0).values.reshape(-1, 1)
        df_col_scaled = scaler.transform(df_col_temp)
        df_col_scaled[nan_mask] = 0.0
        df_col_scaled_clipped = np.clip(df_col_scaled, a_min=0.0, a_max=1.0)
        return df_col_scaled_clipped.ravel(), scaler

    if le_mcc is None:
        df['mcc_encoded'], le_mcc = fit_or_transform_label(None, df['mcc'], fit=True)
        df['zip_encoded'], le_zip = fit_or_transform_label(None, df['zip'], fit=True)
        df['merchant_region_encoded'], le_region = fit_or_transform_label(None, df['merchant_region'], fit=True)
        df['use_chip_encoded'], le_use_chip = fit_or_transform_label(None, df['use_chip'], fit=True)
        scalers = {}
        numerical_cols_to_scale = ['delta_time', 'amount', 'delta_distance', 'account_age_days', 'months_to_expiry']
        for col in numerical_cols_to_scale:
            if col == 'amount':
                df[col] = np.log1p(df[col])
            df[col], scalers[col] = fit_or_transform_scaler(None, df[col], fit=True)
    else:
        df['mcc_encoded'], _ = fit_or_transform_label(le_mcc, df['mcc'], fit=False)
        df['zip_encoded'], _ = fit_or_transform_label(le_zip, df['zip'], fit=False)
        df['merchant_region_encoded'], _ = fit_or_transform_label(le_region, df['merchant_region'], fit=False)
        df['use_chip_encoded'], _ = fit_or_transform_label(le_use_chip, df['use_chip'], fit=False)
        numerical_cols_to_scale = ['delta_time', 'amount', 'delta_distance', 'account_age_days', 'months_to_expiry']
        for col in numerical_cols_to_scale:
            if col == 'amount':
                df[col] = np.log1p(df[col])
            df[col], _ = fit_or_transform_scaler(scalers[col], df[col], fit=False)

    return df, le_mcc, le_zip, le_region, le_use_chip, scalers


# 2. 거래 기준 과거 24시간 이내 거래를 포함하는 윈도우 생성
def create_time_based_sequences(df, meta_cols, numerical_cols, categorical_cols, max_seq_len=20,
                                time_window=timedelta(days=1)):
    sequences_num, sequences_cat, masks, labels, window_meta = [], [], [], [], []
    for (cid, card), g in tqdm(df.groupby(['client_id', 'card_id']), desc="Sequence 생성", ncols=90):
        g = g.sort_values('date').reset_index(drop=True)
        features_num = g[numerical_cols].values.astype(np.float32)
        features_cat = g[categorical_cols].values.astype(np.float32)
        labels_seq = g['fraud'].values
        length = len(g)
        for end_idx in range(length):
            window_end_date = g['date'].iloc[end_idx]
            window_start_date = window_end_date - time_window
            window_df = g[(g['date'] > window_start_date) & (g['date'] <= window_end_date)]
            seq_len = len(window_df)
            if seq_len == 0: continue
            if seq_len < max_seq_len:
                features_num_window = features_num[window_df.index]
                features_cat_window = features_cat[window_df.index]
                pad_len = max_seq_len - seq_len
                seq_num = np.vstack([np.zeros((pad_len, len(numerical_cols)), dtype=np.float32), features_num_window])
                seq_cat = np.vstack(
                    [np.full((pad_len, len(categorical_cols)), 0, dtype=np.float32), features_cat_window])
                mask = np.array([True] * pad_len + [False] * seq_len)
                meta_window_df = window_df
            else:
                features_num_window = features_num[window_df.index][-max_seq_len:]
                features_cat_window = features_cat[window_df.index][-max_seq_len:]
                seq_len = max_seq_len
                pad_len = 0
                seq_num = features_num_window
                seq_cat = features_cat_window
                mask = np.array([False] * max_seq_len)
                meta_window_df = window_df.iloc[-max_seq_len:]
            lbl = labels_seq[end_idx]
            meta_dict = {
                'client_id': cid, 'card_id': card,
                'window_start_date': meta_window_df['date'].iloc[0],
                'window_end_date': meta_window_df['date'].iloc[-1],
                'num_transactions_in_window': seq_len,
            }
            last_real_idx = meta_window_df.index[-1]
            for mc in meta_cols:
                meta_dict['last_' + mc] = g[mc].iloc[last_real_idx]
            window_meta.append(meta_dict)
            sequences_num.append(seq_num)
            sequences_cat.append(seq_cat)
            masks.append(mask)
            labels.append(lbl)
    sequences_num = np.array(sequences_num, dtype=np.float32)
    sequences_cat = np.array(sequences_cat, dtype=np.float32)
    masks = np.array(masks, dtype=bool)
    labels = np.array(labels, dtype=np.float32)
    return sequences_num, sequences_cat, masks, labels, pd.DataFrame(window_meta)


# 3. 고객별 시계열 및 사기 거래 블록 보존 분리
def split_sequences_with_fraud_block(sequences_num, sequences_cat, masks, labels, meta_df, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    meta_df['seq_index'] = np.arange(len(meta_df))
    meta_df['last_fraud'] = labels[:]
    train_indices, val_indices = [], []
    for client_id, group in meta_df.groupby('client_id'):
        group = group.sort_values('window_end_date').reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue
        fraud_arr = group['last_fraud'].values
        fraud_blocks = []
        start_idx = None
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
    X_train_num = sequences_num[train_indices]
    X_val_num = sequences_num[val_indices]
    X_train_cat = sequences_cat[train_indices]
    X_val_cat = sequences_cat[val_indices]
    mask_train = masks[train_indices]
    mask_val = masks[val_indices]
    labels_train = labels[train_indices]
    labels_val = labels[val_indices]
    return X_train_num, X_train_cat, X_val_num, X_val_cat, mask_train, mask_val, labels_train, labels_val


# 4. Transformer 모델
class TransactionTransformer(nn.Module):
    def __init__(self, numerical_dim, mcc_vocab_size, zip_vocab_size, region_vocab_size, use_chip_vocab_size,
                 model_dim=64, num_heads=4, num_layers=2, embedding_dim=32, num_classes=2):
        super().__init__()
        self.mcc_emb = nn.Embedding(mcc_vocab_size + 1, 16, padding_idx=0)
        self.zip_emb = nn.Embedding(zip_vocab_size + 1, 16, padding_idx=0)
        self.region_emb = nn.Embedding(region_vocab_size + 1, 8, padding_idx=0)
        self.use_chip_emb = nn.Embedding(use_chip_vocab_size + 1, 4, padding_idx=0)
        self.input_dim_adjusted = numerical_dim + 16 + 16 + 8 + 4
        self.embed = nn.Linear(self.input_dim_adjusted, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Linear(model_dim, embedding_dim)
        self.classification_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x_num, x_cat_mcc, x_cat_zip, x_cat_region, x_cat_use_chip, mask=None):
        mcc_emb = self.mcc_emb(x_cat_mcc.long())
        zip_emb = self.zip_emb(x_cat_zip.long())
        region_emb = self.region_emb(x_cat_region.long())
        use_chip_emb = self.use_chip_emb(x_cat_use_chip.long())
        x = torch.cat([x_num, mcc_emb, zip_emb, region_emb, use_chip_emb], dim=-1)
        x = self.embed(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        masked_output = x * (~mask).unsqueeze(-1)
        sequence_lengths = (~mask).sum(dim=1).unsqueeze(-1)
        sum_pooling = masked_output.sum(dim=1)
        avg_pooling = sum_pooling / sequence_lengths.clamp(min=1)
        embedding = self.embedding_head(avg_pooling)
        logits = self.classification_head(embedding)
        return embedding, logits


# 5. EarlyStopping 클래스
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.best_epoch = epoch
            print(f"New best validation loss: {self.best_loss:.4f}")
        elif val_loss <= self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            self.best_epoch = epoch
            print(f"New best validation loss: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# 6. 모델 추론 및 임베딩 생성
def get_predictions_and_embeddings(model, sequences_num, sequences_cat, masks, labels, batch_size=2048):
    model.eval()
    all_embeddings = []
    all_logits = []
    all_labels = []

    X_num_tensor = torch.tensor(sequences_num, dtype=torch.float32).to(device)
    X_cat_tensor = torch.tensor(sequences_cat, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(masks, dtype=torch.bool).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences_num), batch_size), desc="모델 추론 및 임베딩 생성", ncols=90):
            batch_X_num = X_num_tensor[i:i + batch_size]
            batch_X_cat = X_cat_tensor[i:i + batch_size]
            batch_mask = mask_tensor[i:i + batch_size]
            batch_y = labels_tensor[i:i + batch_size]

            embeddings, logits = model(
                batch_X_num,
                batch_X_cat[..., 0],
                batch_X_cat[..., 1],
                batch_X_cat[..., 2],
                batch_X_cat[..., 3],
                mask=batch_mask
            )

            all_embeddings.append(embeddings.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(batch_y.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    return embeddings_tensor, logits_tensor, labels_tensor


# 7. 라벨 인코더 및 스케일러 저장 및 불러오기
def save_objects(objects, path):
    with open(path, 'wb') as f:
        pickle.dump(objects, f)
    print(f"[INFO] 객체 저장 완료: {path}")


def load_objects(path):
    with open(path, 'rb') as f:
        objects = pickle.load(f)
    print(f"[INFO] 객체 불러오기 완료: {path}")
    return objects


if __name__ == "__main__":

    # 파일 경로 설정
    file_paths = {
        'train_data_input': './data/train_transactions_Clean.csv',
        'test_data_input': './data/test_transactions_Clean.csv',
        'train_embeddings_output': './embedding_result/transformer_train_embeddings.csv',
        'test_embeddings_output': './embedding_result/transformer_test_embeddings.csv',
        'train_meta_output': './embedding_result/transformer_train_window_meta.csv',
        'test_meta_output': './embedding_result/transformer_test_window_meta.csv',
        'model_output': './trained_model/transformer_model.pth',
        'label_encoder_output': './scaler_and_encoder/transformer_label_encoders.pkl',
        'scalers_output': './scaler_and_encoder/transformer_scalers.pkl',
        'transformer_test_results_output': './npz/transformer_test_results.npz'
    }

    numerical_feature_cols = [
        'delta_time', 'amount', 'is_online', 'delta_distance',
        'months_to_expiry', 'account_age_days',
        'hour', 'dayofweek', 'month', 'is_weekend', 'is_night',
        'is_delta_time_missing', 'is_delta_distance_missing'
    ]
    categorical_feature_cols = [
        'mcc_encoded', 'zip_encoded', 'merchant_region_encoded', 'use_chip_encoded'
    ]
    meta_cols = [
        'amount', 'merchant_id', 'merchant_city', 'merchant_state',
        'zip', 'mcc', 'use_chip', 'fraud'
    ]

    # 모델 학습 및 저장 단계
    if not os.path.exists(file_paths['model_output']):
        print("[STEP 1] 학습용 데이터로 Transformer 모델 학습 및 저장...")
        df_train, le_mcc, le_zip, le_region, le_use_chip, scalers = preprocess_and_feature_engineer(
            pd.read_csv(file_paths['train_data_input'], dtype={'zip': str})
        )
        save_objects({'le_mcc': le_mcc, 'le_zip': le_zip, 'le_region': le_region, 'le_use_chip': le_use_chip},
                     file_paths['label_encoder_output'])
        save_objects(scalers, file_paths['scalers_output'])

        sequences_num_train, sequences_cat_train, masks_train, labels_train, window_meta_df_train = create_time_based_sequences(
            df_train, meta_cols, numerical_feature_cols, categorical_feature_cols
        )

        X_train_num, X_train_cat, X_val_num, X_val_cat, mask_train, mask_val, labels_train, labels_val = split_sequences_with_fraud_block(
            sequences_num_train, sequences_cat_train, masks_train, labels_train, window_meta_df_train, train_ratio=0.8
        )

        X_train_num_tensor = torch.tensor(X_train_num, dtype=torch.float32).to(device)
        X_train_cat_tensor = torch.tensor(X_train_cat, dtype=torch.float32).to(device)
        mask_train_tensor = torch.tensor(mask_train, dtype=torch.bool).to(device)
        labels_train_tensor = torch.tensor(labels_train, dtype=torch.long).to(device)

        X_val_num_tensor = torch.tensor(X_val_num, dtype=torch.float32).to(device)
        X_val_cat_tensor = torch.tensor(X_val_cat, dtype=torch.float32).to(device)
        mask_val_tensor = torch.tensor(mask_val, dtype=torch.bool).to(device)
        labels_val_tensor = torch.tensor(labels_val, dtype=torch.long).to(device)

        print(f"Total train sequences: {len(sequences_num_train)}")
        print(f"Training sequences: {len(X_train_num)}")
        print(f"Validation sequences: {len(X_val_num)}")

        model = TransactionTransformer(
            numerical_dim=len(numerical_feature_cols),
            mcc_vocab_size=len(le_mcc.classes_),
            zip_vocab_size=len(le_zip.classes_),
            region_vocab_size=len(le_region.classes_),
            use_chip_vocab_size=len(le_use_chip.classes_)
        ).to(device)

        class_counts = np.bincount(labels_train.astype(int))
        weight = torch.tensor(1. / class_counts, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        num_epochs = 200
        batch_size = 2048

        print("\n[INFO] Transformer 학습 시작...")
        for epoch in range(1, num_epochs + 1):
            model.train()
            total_train_loss = 0
            num_train_batches = int(np.ceil(len(X_train_num) / batch_size))
            train_indices = np.arange(len(X_train_num))
            np.random.shuffle(train_indices)
            for i in tqdm(range(num_train_batches), desc=f"Epoch {epoch} Training", ncols=90):
                batch_indices = train_indices[i * batch_size: (i + 1) * batch_size]
                batch_X_num = X_train_num_tensor[batch_indices]
                batch_X_cat = X_train_cat_tensor[batch_indices]
                batch_mask = mask_train_tensor[batch_indices]
                batch_y = labels_train_tensor[batch_indices]
                optimizer.zero_grad()
                _, logits = model(
                    batch_X_num,
                    batch_X_cat[..., 0],
                    batch_X_cat[..., 1],
                    batch_X_cat[..., 2],
                    batch_X_cat[..., 3],
                    mask=batch_mask
                )
                loss = criterion(logits, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                optimizer.step()
                total_train_loss += loss.item()
            model.eval()
            total_val_loss = 0
            num_val_batches = int(np.ceil(len(X_val_num) / batch_size))
            with torch.no_grad():
                for i in range(num_val_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(X_val_num))
                    batch_X_num = X_val_num_tensor[start_idx:end_idx]
                    batch_X_cat = X_val_cat_tensor[start_idx:end_idx]
                    batch_mask = mask_val_tensor[start_idx:end_idx]
                    batch_y = labels_val_tensor[start_idx:end_idx]
                    _, logits = model(
                        batch_X_num,
                        batch_X_cat[..., 0],
                        batch_X_cat[..., 1],
                        batch_X_cat[..., 2],
                        batch_X_cat[..., 3],
                        mask=batch_mask
                    )
                    loss = criterion(logits, batch_y)
                    total_val_loss += loss.item()
            val_loss = total_val_loss / num_val_batches
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            print(
                f"Epoch {epoch}/{num_epochs}, Train Loss: {total_train_loss / num_train_batches:.4f}, Val Loss: {val_loss:.4f}")
        model.load_state_dict(early_stopping.best_model_state)

        torch.save(model.state_dict(), file_paths['model_output'])
        print(f"\n[INFO] 학습 완료된 모델 저장 완료: {file_paths['model_output']}")

    else:
        print("[STEP 1] 저장된 Transformer 모델 불러오기...")
        encoders = load_objects(file_paths['label_encoder_output'])
        scalers = load_objects(file_paths['scalers_output'])
        le_mcc, le_zip, le_region, le_use_chip = encoders['le_mcc'], encoders['le_zip'], encoders['le_region'], \
            encoders['le_use_chip']

        model = TransactionTransformer(
            numerical_dim=len(numerical_feature_cols),
            mcc_vocab_size=len(le_mcc.classes_),
            zip_vocab_size=len(le_zip.classes_),
            region_vocab_size=len(le_region.classes_),
            use_chip_vocab_size=len(le_use_chip.classes_)
        ).to(device)

        model.load_state_dict(torch.load(file_paths['model_output'], map_location=device))
        print(f"[INFO] 모델 로드 완료: {file_paths['model_output']}")

    print("\n[STEP 2] 전체 학습용 및 테스트용 데이터에 대한 임베딩 생성 및 성능 평가...")
    model.eval()

    # 학습용 데이터 처리 (임베딩 생성)
    df_train_raw = pd.read_csv(file_paths['train_data_input'], dtype={'zip': str})
    df_train_full, _, _, _, _, _ = preprocess_and_feature_engineer(
        df_train_raw, le_mcc, le_zip, le_region, le_use_chip, scalers
    )
    sequences_num_train_full, sequences_cat_train_full, masks_train_full, labels_train_full, window_meta_df_train_full = create_time_based_sequences(
        df_train_full, meta_cols, numerical_feature_cols, categorical_feature_cols
    )

    embeddings_train_tensor, _, _ = get_predictions_and_embeddings(
        model, sequences_num_train_full, sequences_cat_train_full, masks_train_full, labels_train_full
    )
    embeddings_train_df = pd.DataFrame(embeddings_train_tensor.numpy())
    embeddings_train_df.insert(0, 'index', window_meta_df_train_full.index)
    embeddings_train_df.to_csv(file_paths['train_embeddings_output'], index=False)
    window_meta_df_train_full.to_csv(file_paths['train_meta_output'], index=False)
    print(f"[INFO] 학습용 임베딩 및 메타데이터 저장 완료: {file_paths['train_embeddings_output']}")

    # 테스트용 데이터 처리 (임베딩 생성)
    df_test_raw = pd.read_csv(file_paths['test_data_input'], dtype={'zip': str})
    df_test_full, _, _, _, _, _ = preprocess_and_feature_engineer(
        df_test_raw, le_mcc, le_zip, le_region, le_use_chip, scalers
    )
    sequences_num_test_full, sequences_cat_test_full, masks_test_full, labels_test_full, window_meta_df_test_full = create_time_based_sequences(
        df_test_full, meta_cols, numerical_feature_cols, categorical_feature_cols
    )

    embeddings_test_tensor, logits_test_tensor, labels_test_tensor = get_predictions_and_embeddings(
        model, sequences_num_test_full, sequences_cat_test_full, masks_test_full, labels_test_full
    )

    # 성능 지표 계산
    probs_test = torch.softmax(logits_test_tensor, dim=1)[:, 1].numpy()
    labels_test = labels_test_tensor.numpy()

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in thresholds:
        preds = (probs_test >= threshold).astype(int)
        f1 = f1_score(labels_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    preds_best_f1 = (probs_test >= best_threshold).astype(int)

    print(f"\n--- Transformer 모델 최종 성능 평가 결과 (테스트 데이터셋) ---")
    print(f"Optimal Threshold: {best_threshold:.2f} with F1-Score: {best_f1:.4f}")
    print(f"Accuracy: {accuracy_score(labels_test, preds_best_f1):.4f}")
    print(
        f"ROC-AUC: {roc_auc_score(labels_test, probs_test):.4f} | PR-AUC: {average_precision_score(labels_test, probs_test):.4f}")
    print(classification_report(labels_test, preds_best_f1, digits=4))
    print("혼동 행렬:\n", confusion_matrix(labels_test, preds_best_f1))

    # 라벨 및 예측 결과 numpy 파일 저장
    np.savez_compressed(
        file_paths['transformer_test_results_output'],
        labels=labels_test,
        probs=probs_test
    )
    print(f"[INFO] 테스트 결과(예측 확률 및 실제 레이블) 저장 완료: {file_paths['transformer_test_results_output']}")

    # 임베딩 저장
    embeddings_test_df = pd.DataFrame(embeddings_test_tensor.numpy())
    embeddings_test_df.insert(0, 'index', window_meta_df_test_full.index)
    embeddings_test_df.to_csv(file_paths['test_embeddings_output'], index=False)
    window_meta_df_test_full.to_csv(file_paths['test_meta_output'], index=False)
    print(f"[INFO] 테스트용 임베딩 및 메타데이터 저장 완료: {file_paths['test_embeddings_output']}")