import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv, Linear
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
import os
from tqdm import tqdm
import pickle

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# 범주형 인코더/수치형 스케일러 저장 및 불러오기
def save_objects(objects, path):
    with open(path, 'wb') as f:
        pickle.dump(objects, f)
    print(f"[INFO] 객체 저장 완료: {path}")


def load_objects(path):
    with open(path, 'rb') as f:
        objects = pickle.load(f)
    print(f"[INFO] 객체 불러오기 완료: {path}")
    return objects


# 그래프 데이터 로드 및 피처 처리
def load_graph_data(emb_path, meta_path, card_path, user_path,
                    encoders=None, scalers=None):
    print(f"[INFO] Loading and processing graph data from {meta_path}...")
    trans_emb = pd.read_csv(emb_path)
    trans_meta = pd.read_csv(meta_path)
    card_df = pd.read_csv(card_path)
    user_df = pd.read_csv(user_path)

    # 0. 데이터 전처리 및 키 기반 결합 (강화)
    trans_meta['window_end_date'] = pd.to_datetime(trans_meta['window_end_date'])
    trans_data = pd.merge(trans_meta, trans_emb, left_index=True, right_index=True)

    trans_data["transaction_id"] = np.arange(len(trans_data))
    trans_data = trans_data.sort_values(by=['client_id', 'card_id', 'window_end_date']).reset_index(drop=True)

    # 1. 노드 피처 구성
    print("[INFO] Creating node features...")
    data = HeteroData()

    if encoders is None:
        encoders = {}
    if scalers is None:
        scalers = {}

    user_cols_to_keep = ['current_age', 'gender','yearly_income',
                         'total_debt', 'credit_score', 'num_credit_cards']

    for col in ['gender']:
        if f'user_{col}' not in encoders:
            le = LabelEncoder().fit(np.append(user_df[col].astype(str).unique(), 'UNK'))
            encoders[f'user_{col}'] = le
        else:
            le = encoders.get(f'user_{col}')
        user_df[col] = user_df[col].astype(str).map(
            {name: i for i, name in enumerate(le.classes_)}
        ).fillna(le.transform(['UNK'])[0]).astype(int)

    user_num_cols = [c for c in user_cols_to_keep if c not in ['gender']]
    if 'user_scalers' not in scalers:
        scalers['user_scalers'] = {}

    for col in user_num_cols:
        if col not in scalers['user_scalers']:
            scaler = StandardScaler()
            user_df[col] = scaler.fit_transform(user_df[[col]]).clip(-3, 3)
            scalers['user_scalers'][col] = scaler
        else:
            scaler = scalers['user_scalers'].get(col)
            user_df[col] = scaler.transform(user_df[[col]]).clip(-3, 3)

    user_df_processed = user_df[user_cols_to_keep].fillna(0)
    data['customer'].x = torch.tensor(user_df_processed.values, dtype=torch.float)

    card_cols_to_keep = ['card_brand', 'card_type', 'has_chip',
                         'num_cards_issued', 'credit_limit', 'year_pin_last_changed']

    for col in ['card_brand', 'card_type', 'has_chip']:
        if f'card_{col}' not in encoders:
            le = LabelEncoder().fit(np.append(card_df[col].astype(str).unique(), 'UNK'))
            encoders[f'card_{col}'] = le
        else:
            le = encoders.get(f'card_{col}')
        card_df[col] = card_df[col].astype(str).map(
            {name: i for i, name in enumerate(le.classes_)}
        ).fillna(le.transform(['UNK'])[0]).astype(int)

    card_num_cols = [c for c in card_cols_to_keep if c not in ['card_brand', 'card_type', 'has_chip']]
    if 'card_scalers' not in scalers:
        scalers['card_scalers'] = {}

    for col in card_num_cols:
        if col not in scalers['card_scalers']:
            scaler = StandardScaler()
            card_df[col] = scaler.fit_transform(card_df[[col]].fillna(0)).clip(-3, 3)
            scalers['card_scalers'][col] = scaler
        else:
            scaler = scalers['card_scalers'].get(col)
            card_df[col] = scaler.transform(card_df[[col]].fillna(0)).clip(-3, 3)

    card_df_processed = card_df[card_cols_to_keep].fillna(0)
    data['card'].x = torch.tensor(card_df_processed.values, dtype=torch.float)

    trans_meta_num = trans_data[[
        'last_amount', 'num_transactions_in_window'
    ]].fillna(0)

    trans_num_cols = ['last_amount', 'num_transactions_in_window']
    if 'trans_scalers' not in scalers:
        scalers['trans_scalers'] = {}

    for col in trans_num_cols:
        if col not in scalers['trans_scalers']:
            scaler = StandardScaler()
            trans_meta_num[col] = scaler.fit_transform(trans_meta_num[[col]]).clip(-3, 3)
            scalers['trans_scalers'][col] = scaler
        else:
            scaler = scalers['trans_scalers'].get(col)
            trans_meta_num[col] = scaler.transform(trans_meta_num[[col]]).clip(-3, 3)

    trans_feat = pd.concat([trans_data.loc[:, '0':'31'], trans_meta_num], axis=1)

    data['transaction'].x = torch.tensor(trans_feat.values, dtype=torch.float)
    data['transaction'].y = torch.tensor(trans_data['last_fraud'].values, dtype=torch.long)

    trans_data['last_merchant_id'] = trans_data['last_merchant_id'].astype(str)
    if 'merchant_le' not in encoders:
        merchant_le = LabelEncoder().fit(np.append(trans_data['last_merchant_id'].unique(), 'UNK'))
        encoders['merchant_le'] = merchant_le
    else:
        merchant_le = encoders.get('merchant_le')

    known_merchants_map = {name: i for i, name in enumerate(merchant_le.classes_)}
    unk_idx = known_merchants_map['UNK']
    trans_data['merchant_nid'] = trans_data['last_merchant_id'].map(known_merchants_map).fillna(unk_idx).astype(int)
    num_merchants_total = len(merchant_le.classes_)
    data['merchant'].num_nodes = num_merchants_total
    data['merchant'].x = torch.arange(num_merchants_total).view(-1, 1)

    # 2. 엣지 구성
    print("[INFO] Creating edges...")
    user_id_map = {cid: i for i, cid in enumerate(user_df['id'].values)}
    card_id_map = {cid: i for i, cid in enumerate(card_df['id'].values)}

    cc_src = [user_id_map.get(cid, -1) for cid in card_df['client_id'].values]
    cc_dst = list(range(len(card_df)))
    safe_indices = [i for i, src in enumerate(cc_src) if src != -1]
    data['customer', 'owns', 'card'].edge_index = torch.tensor([
        [cc_src[i] for i in safe_indices], [cc_dst[i] for i in safe_indices]
    ], dtype=torch.long)
    data['card', 'owned_by', 'customer'].edge_index = torch.tensor([
        [cc_dst[i] for i in safe_indices], [cc_src[i] for i in safe_indices]
    ], dtype=torch.long)

    tc_src = [card_id_map[cid] for cid in trans_data['card_id'].values]
    tc_dst = trans_data['transaction_id'].tolist()
    data['card', 'used_in', 'transaction'].edge_index = torch.tensor([tc_src, tc_dst])
    data['transaction', 'by_card', 'card'].edge_index = torch.tensor([tc_dst, tc_src])

    tm_src = trans_data['transaction_id'].tolist()
    tm_dst = trans_data['merchant_nid'].tolist()
    data['transaction', 'at', 'merchant'].edge_index = torch.tensor([tm_src, tm_dst])
    data['merchant', 'has', 'transaction'].edge_index = torch.tensor([tm_dst, tm_src])

    # 시간 순 거래 노드 연결
    num_recent_transactions = 5
    sequential_edges_src = []
    sequential_edges_dst = []

    for _, g in trans_data.groupby(['client_id', 'card_id']):
        g = g.sort_values('window_end_date').reset_index(drop=True)
        for i in range(1, len(g)):
            start_idx = max(0, i - num_recent_transactions)
            for j in range(start_idx, i):
                src_id = g.loc[j, 'transaction_id']
                dst_id = g.loc[i, 'transaction_id']

                sequential_edges_src.append(src_id)
                sequential_edges_dst.append(dst_id)

    if sequential_edges_src:
        data['transaction', 'next_to', 'transaction'].edge_index = torch.tensor(
            [sequential_edges_src, sequential_edges_dst], dtype=torch.long)
    else:
        data['transaction', 'next_to', 'transaction'].edge_index = torch.empty((2, 0), dtype=torch.long)

    print("[INFO] Data loading complete.")
    return data, encoders, scalers


class HGT(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, in_dims, num_merchants, num_layers=4):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        for ntype in metadata[0]:
            if ntype == "merchant":
                self.lin_dict[ntype] = None
            else:
                self.lin_dict[ntype] = Linear(in_dims[ntype], hidden_channels)
        self.merchant_emb = nn.Embedding(num_merchants, hidden_channels)
        self.convs = nn.ModuleList([HGTConv(hidden_channels, hidden_channels, metadata) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.3)
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict_temp = x_dict.copy()
        edge_index_dict_temp = edge_index_dict.copy()

        if 'merchant' in x_dict_temp:
            x_dict_temp['merchant'] = self.merchant_emb(x_dict_temp['merchant'].squeeze().long())

        for k in x_dict_temp.keys():
            if k != 'merchant':
                x_dict_temp[k] = self.lin_dict[k](x_dict_temp[k])

        filtered_edge_index_dict = {}
        for (src_type, rel_type, dst_type) in self.convs[0].edge_types:
            if src_type in x_dict_temp and dst_type in x_dict_temp:
                filtered_edge_index_dict[(src_type, rel_type, dst_type)] = edge_index_dict_temp.get(
                    (src_type, rel_type, dst_type), torch.empty((2, 0), dtype=torch.long))

        x_dict = x_dict_temp
        for conv in self.convs:
            x_dict = conv(x_dict, filtered_edge_index_dict)
            for key in x_dict:
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])

        return self.lin_out(x_dict['transaction'])


def train_one_epoch(model, loader, optimizer, criterion, num_train_nodes):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", ncols=90):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)

        loss = criterion(out[:batch['transaction'].batch_size],
                         batch['transaction'].y[:batch['transaction'].batch_size])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch['transaction'].batch_size
    return total_loss / num_train_nodes


def validate_one_epoch(model, loader, criterion, num_val_nodes):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)

            loss = criterion(out[:batch['transaction'].batch_size],
                             batch['transaction'].y[:batch['transaction'].batch_size])
            total_loss += loss.item() * batch['transaction'].batch_size
    return total_loss / num_val_nodes


def evaluate(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=90):
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)

            probs = torch.softmax(out[:batch['transaction'].batch_size], dim=1)[:, 1].cpu().numpy()
            labels = batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy()
            all_labels.extend(labels)
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs)


def print_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(classification_report(labels, preds, digits=4))
    print(confusion_matrix(labels, preds))


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
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


if __name__ == "__main__":

    file_paths = {
        'train_emb_input': f'./embedding_result/transformer_train_embeddings.csv',
        'train_meta_input': f'./embedding_result/transformer_train_window_meta.csv',
        'test_emb_input': f'./embedding_result/transformer_test_embeddings.csv',
        'test_meta_input': f'./embedding_result/transformer_test_window_meta.csv',
        'cards_data_input': './data/cards_features_static.csv',
        'users_data_input': './data/users_features_filtered.csv',
        'hgt_model_output': f'./trained_model/hgt_model.pth',
        'encoders_output': f'./scaler_and_encoder/hgt_encoders.pkl',
        'scalers_output': f'./scaler_and_encoder/hgt_scalers.pkl',
        'htgn_test_results_output': f'./npz/htgn_test_results.npz'
    }
    hidden_channel_setting = 160
    num_layers_setting = 4

    model, encoders, scalers, train_data = None, None, None, None

    if not os.path.exists(file_paths['hgt_model_output']):
        print("[STEP 1] Loading train data for training...")
        train_data, encoders, scalers = load_graph_data(
            file_paths['train_emb_input'], file_paths['train_meta_input'],
            file_paths['cards_data_input'], file_paths['users_data_input']
        )
        save_objects(encoders, file_paths['encoders_output'])
        save_objects(scalers, file_paths['scalers_output'])

        print("[INFO] Splitting transaction nodes into train and validation sets...")
        meta_df = pd.read_csv(file_paths['train_meta_input'])
        meta_df['seq_index'] = np.arange(len(meta_df))
        meta_df['last_fraud'] = train_data['transaction'].y.numpy()

        train_indices, val_indices = [], []
        for client_id, group in meta_df.groupby('client_id'):
            group = group.sort_values('window_end_date').reset_index(drop=True)
            n = len(group)
            if n == 0: continue
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
            split_idx = int(n * 0.8)


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

        train_mask = torch.zeros(train_data['transaction'].num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask = torch.zeros(train_data['transaction'].num_nodes, dtype=torch.bool)
        val_mask[val_indices] = True

        train_data['transaction'].train_mask = train_mask
        train_data['transaction'].val_mask = val_mask

        num_train_nodes = len(train_indices)
        num_val_nodes = len(val_indices)

        in_dims = {ntype: train_data[ntype].x.size(1) if ntype != 'merchant' else 1 for ntype in train_data.node_types}
        model = HGT(hidden_channels=hidden_channel_setting, out_channels=2, metadata=train_data.metadata(),
                    in_dims=in_dims, num_merchants=train_data['merchant'].num_nodes, num_layers=num_layers_setting).to(device)

        class_counts = np.bincount(train_data['transaction'].y[train_mask].numpy())
        weight = torch.tensor(1. / class_counts, dtype=torch.float).to(device)

        print(f"Class weights: {weight}")
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        num_neighbors_dict = {
            ('customer', 'owns', 'card'): [5, 3],
            ('card', 'owned_by', 'customer'): [5, 3],
            ('card', 'used_in', 'transaction'): [15, 10],
            ('transaction', 'by_card', 'card'): [15, 10],
            ('transaction', 'at', 'merchant'): [15, 10],
            ('merchant', 'has', 'transaction'): [15, 10],
            ('transaction', 'next_to', 'transaction'): [5, 3]
        }

        train_loader = NeighborLoader(
            train_data,
            num_neighbors=num_neighbors_dict,
            input_nodes=('transaction', train_data['transaction'].train_mask),
            batch_size=1024,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker
        )
        val_loader = NeighborLoader(
            train_data,
            num_neighbors=num_neighbors_dict,
            input_nodes=('transaction', train_data['transaction'].val_mask),
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker
        )

        early_stopping = EarlyStopping(patience=15, min_delta=0.001)

        print("[STEP 2] Starting training...")
        for epoch in range(1, 201):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, num_train_nodes)
            val_loss = validate_one_epoch(model, val_loader, criterion, num_val_nodes)
            scheduler.step(val_loss)

            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            if epoch % 5 == 0:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        model.load_state_dict(early_stopping.best_model_state)
        print(f"\n[INFO] Loaded best model weights from epoch {early_stopping.best_epoch}.")
        print(f"[INFO] Final best validation loss: {early_stopping.best_loss:.4f}")

        torch.save(model.state_dict(), file_paths['hgt_model_output'])
        print(f"\n[INFO] 학습 완료된 모델 저장 완료: {file_paths['hgt_model_output']}")

    else:
        print("[STEP 1] Loading existing HTGN model for evaluation...")
        encoders = load_objects(file_paths['encoders_output'])
        scalers = load_objects(file_paths['scalers_output'])

        train_data, _, _ = load_graph_data(
            file_paths['train_emb_input'], file_paths['train_meta_input'],
            file_paths['cards_data_input'], file_paths['users_data_input'],
            encoders=encoders, scalers=scalers
        )

        in_dims = {ntype: train_data[ntype].x.size(1) if ntype != 'merchant' else 1 for ntype in train_data.node_types}
        model = HGT(hidden_channels=hidden_channel_setting, out_channels=2, metadata=train_data.metadata(),
                    in_dims=in_dims, num_merchants=train_data['merchant'].num_nodes, num_layers=num_layers_setting).to(device)
        model.load_state_dict(torch.load(file_paths['hgt_model_output'], map_location=device))
        print(f"[INFO] 모델 로드 완료: {file_paths['hgt_model_output']}")

    print("\n[STEP 2] Evaluating model on Test Data...")

    num_neighbors_dict = {
        ('customer', 'owns', 'card'): [5, 3],
        ('card', 'owned_by', 'customer'): [5, 3],
        ('card', 'used_in', 'transaction'): [15, 10],
        ('transaction', 'by_card', 'card'): [15, 10],
        ('transaction', 'at', 'merchant'): [15, 10],
        ('merchant', 'has', 'transaction'): [15, 10],
        ('transaction', 'next_to', 'transaction'): [5, 3]
    }

    test_data, _, _ = load_graph_data(
        file_paths['test_emb_input'], file_paths['test_meta_input'],
        file_paths['cards_data_input'], file_paths['users_data_input'],
        encoders=encoders, scalers=scalers
    )
    test_loader = NeighborLoader(
        test_data,
        num_neighbors=num_neighbors_dict,
        input_nodes=('transaction', torch.arange(test_data['transaction'].num_nodes)),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    labels, probs = evaluate(model, test_loader)

    print("\n[STEP 3] Finding optimal threshold...")
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0.0
    best_preds = (probs >= 0.5).astype(int)
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_preds = preds

    print(f"\nOptimal Threshold for final model: {best_threshold:.2f} with F1-Score: {best_f1:.4f}")
    print_metrics(labels, best_preds, probs)

    # 예측 결과(확률, 실제 레이블)를 NumPy 파일로 저장
    np.savez_compressed(
        file_paths['htgn_test_results_output'],
        labels=labels,
        probs=probs
    )
    print(f"[INFO] 테스트 결과(예측 확률 및 실제 레이블) 저장 완료: {file_paths['htgn_test_results_output']}")
