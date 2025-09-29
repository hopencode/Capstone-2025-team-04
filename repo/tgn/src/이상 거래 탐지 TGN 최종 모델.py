import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt
import sys
import os
# 현재 파일 기준 상위 폴더로 이동 후 src/tgn 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "tgn"))

# sys.path.append('/Users/hyeon/graduation_pjt_workstation/tgn')  # tgn 폴더 경로
from model.tgn import TGN
from utils.utils import NeighborFinder
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from temporal_graph_builder import build_tgn_event_stream_from_window_embeddings

train_meta = "./data/트랜스포머 임베딩 결과/학습용 데이터 학습용 검증용 분할/transformer_train_window_meta_80_V22.csv"
train_emb  = "./data/트랜스포머 임베딩 결과/학습용 데이터 학습용 검증용 분할/transformer_train_embeddings_80_V22.csv"

val_meta   = "./data/트랜스포머 임베딩 결과/학습용 데이터 학습용 검증용 분할/transformer_train_window_meta_20_V22.csv"
val_emb    = "./data/트랜스포머 임베딩 결과/학습용 데이터 학습용 검증용 분할/transformer_train_embeddings_20_V22.csv"

test_meta  = "./data/트랜스포머 임베딩 결과/transformer_test_window_meta_V22.csv"
test_emb   = "./data/트랜스포머 임베딩 결과/transformer_test_embeddings_V22.csv"

tr = pd.read_csv(train_meta, usecols=["card_id", "last_merchant_id"]).astype(str)
vl = pd.read_csv(val_meta,   usecols=["card_id", "last_merchant_id"]).astype(str)
te = pd.read_csv(test_meta,  usecols=["card_id", "last_merchant_id"]).astype(str)

cards   = np.unique(pd.concat([tr["card_id"], vl["card_id"], te["card_id"]], axis=0).values)
merches = np.unique(pd.concat([tr["last_merchant_id"], vl["last_merchant_id"], te["last_merchant_id"]], axis=0).values)

print("총 카드 수:", len(cards))
print("총 상점 수:", len(merches))

card_le_global  = LabelEncoder().fit(cards)
merch_le_global = LabelEncoder().fit(merches)

# 이벤트 스트림 생성
train_events, _, _ = build_tgn_event_stream_from_window_embeddings(
    emb_path=train_emb,
    meta_path=train_meta,
    time_col="window_end_date",
    card_le=card_le_global,
    merch_le=merch_le_global,
    include_zip_city=True,   
)
val_events, _, _ = build_tgn_event_stream_from_window_embeddings(
    emb_path=val_emb,
    meta_path=val_meta,
    time_col="window_end_date",
    card_le=card_le_global,
    merch_le=merch_le_global,
    include_zip_city=True,   
)
test_events,  _, _ = build_tgn_event_stream_from_window_embeddings(
    emb_path=test_emb,
    meta_path=test_meta,
    time_col="window_end_date",
    card_le=card_le_global,
    merch_le=merch_le_global,
    include_zip_city=True,
) 

y_train = train_events.y
pos = (y_train > 0.5).sum()
neg = len(y_train) - pos
pos_weight = torch.tensor([neg / max(pos,1)], dtype=torch.float32, device="cpu")

print(pos_weight)

train = train_events
validation = val_events
test  = test_events

# 시간순 정렬
order_tr = np.argsort(train.t)
order_vl = np.argsort(validation.t)
order_te = np.argsort(test.t)

# --- train arrays ---
train_src = torch.from_numpy(train.src[order_tr]).long().to(device)
train_dst = torch.from_numpy(train.dst[order_tr]).long().to(device)
train_ts  = torch.from_numpy(train.t[order_tr]).float().to(device)      
train_eX  = torch.from_numpy(train.edge_attr[order_tr]).float().to(device)  
train_y   = torch.from_numpy(train.y[order_tr]).float().to(device)      

# --- validation arrays ---
val_src = torch.from_numpy(validation.src[order_vl]).long().to(device)
val_dst = torch.from_numpy(validation.dst[order_vl]).long().to(device)
val_ts  = torch.from_numpy(validation.t[order_vl]).float().to(device)      
val_eX  = torch.from_numpy(validation.edge_attr[order_vl]).float().to(device)  
val_y   = torch.from_numpy(validation.y[order_vl]).float().to(device)     

# --- test arrays ---
test_src = torch.from_numpy(test.src[order_te]).long().to(device)
test_dst = torch.from_numpy(test.dst[order_te]).long().to(device)
test_ts  = torch.from_numpy(test.t[order_te]).float().to(device)
test_eX  = torch.from_numpy(test.edge_attr[order_te]).float().to(device)
test_y   = torch.from_numpy(test.y[order_te]).float().to(device)

num_nodes = train.info["num_nodes"]
num_cards = train.info["num_cards"]
edge_feat_dim = train.info["edge_feat_dim"] 

# train 전용 인접리스트 (시간 포함)
adj_list = [[] for _ in range(num_nodes)]
# edge id는 시간 정렬 후의 인덱스를 그대로 사용
for i in range(len(train_src)):
    s = int(train_src[i].item())
    d = int(train_dst[i].item())
    ts = float(train_ts[i].item())
    adj_list[s].append((d, i, ts))

neighbor_finder = NeighborFinder(adj_list, uniform=True)

# edge feature scaler (train 기준)
edge_scaler = StandardScaler().fit(train_eX.cpu().numpy())
train_eX_scaled = torch.from_numpy(edge_scaler.transform(train_eX.cpu().numpy())).float().to(device)
val_eX_scaled = torch.from_numpy(edge_scaler.transform(val_eX.cpu().numpy())).float().to(device)
test_eX_scaled  = torch.from_numpy(edge_scaler.transform(test_eX.cpu().numpy())).float().to(device) 

train_eX_np = train_eX_scaled.cpu().numpy().astype(np.float32)
val_eX_np = val_eX_scaled.cpu().numpy().astype(np.float32)
test_eX_np  = test_eX_scaled.cpu().numpy().astype(np.float32)
edge_features_all = np.vstack([train_eX_np, val_eX_np, test_eX_np]).astype(np.float32)
np.save("edge_features_all.npy", edge_features_all)

# edge_id 정의
train_edge_ids = torch.arange(len(train_src), device=device)
val_edge_ids   = torch.arange(len(train_src), len(train_src)+len(val_src), device=device)
test_edge_ids  = torch.arange(len(train_src)+len(val_src),
                              len(train_src)+len(val_src)+len(test_src), device=device)

mem_dim = 62

node_features_np = np.zeros((num_nodes, mem_dim), dtype=np.float32)
node_features_np[:num_cards, 0] = 1.0      # is_card 플래그를 첫 번째 차원에
node_features_np[num_cards:, 1] = 1.0      # is_merchant 플래그를 두 번째 차원에

model = TGN(
    neighbor_finder=neighbor_finder,
    node_features=node_features_np,
    edge_features=edge_features_all,
    device=device,
    dropout=0.3,
    use_memory=True, memory_update_at_start=True,
    memory_dimension=mem_dim,  
).to(device) 

pos_weight = pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
edge_input_dim = model.embedding_dimension*2 + edge_feat_dim
fraud_classifier = nn.Linear(edge_input_dim, 1).to(device)
params = list(model.parameters()) + list(fraud_classifier.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=5e-4)

num_epochs = 200
batch_size = 64
best_val = float("inf"); counter=0

train_ts_np = train_ts.cpu().numpy()
val_ts_np   = val_ts.cpu().numpy()

for epoch in range(num_epochs):
    if hasattr(model, "memory") and hasattr(model.memory, "reset_state"):
        model.memory.reset_state()

    # ---------- Train ----------
    model.train()
    epoch_loss = 0.0
    num_batches = int(np.ceil(len(train_src) / batch_size))

    train_iter = tqdm(range(0, len(train_src), batch_size),
                      desc=f"[Train] Epoch {epoch+1}/{num_epochs}", leave=True)

    for start in train_iter:
        b_idx = slice(start, start+batch_size)
        b_src = train_src[b_idx]
        b_dst = train_dst[b_idx]
        b_ts  = train_ts_np[start:start+batch_size].astype(np.float32)
        b_eid = train_edge_ids[b_idx]
        b_y   = train_y[b_idx].float()

        optimizer.zero_grad()
        
        src_emb, dst_emb, _ = model.compute_temporal_embeddings(b_src, b_dst, b_dst.clone(), b_ts, b_eid)

        b_edge_attr = train_eX_scaled[b_idx]
        edge_repr   = torch.cat([src_emb, dst_emb, b_edge_attr], dim=-1)
        logits      = fraud_classifier(edge_repr).view(-1)
        
        loss = criterion(logits, b_y)
        # #----- affinity score-----
        # logits = model.affinity_score(src_emb, dst_emb).view(-1)
        # loss   = criterion(logits, b_y)
        # #----- affinity score-----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += float(loss.item())
        train_iter.set_postfix(curr=float(loss.item()),
                               avg=round(epoch_loss / max(1, (start//batch_size+1)), 4))

    train_loss = epoch_loss / num_batches

    # ---------- Validation ----------
    if hasattr(model, "memory") and hasattr(model.memory, "reset_state"):
        model.memory.reset_state()

    model.eval()
    val_loss = 0.0
    val_batches = int(np.ceil(len(val_src) / batch_size))

    y_buf, p_buf = [], []
    
    val_iter = tqdm(range(0, len(val_src), batch_size),
                    desc=f"[Valid] Epoch {epoch+1}/{num_epochs}", leave=True)

    with torch.no_grad():
        for start in val_iter:
            b_idx = slice(start, start+batch_size)
            v_src = val_src[b_idx]; v_dst = val_dst[b_idx]
            v_ts  = val_ts_np[start:start+batch_size].astype(np.float32)
            v_eid = val_edge_ids[b_idx]
            v_y   = val_y[b_idx].float()

            v_src_emb, v_dst_emb, _ = model.compute_temporal_embeddings(v_src, v_dst, v_dst.clone(), v_ts, v_eid)
            v_edge_attr = val_eX_scaled[b_idx]
            edge_repr   = torch.cat([v_src_emb, v_dst_emb, v_edge_attr], dim=-1)
            v_logits    = fraud_classifier(edge_repr).view(-1)

            val_loss += criterion(v_logits, v_y).item()

            # #----- affinity score-----
            # v_src_emb, v_dst_emb, _ = model.compute_temporal_embeddings(
            #     v_src, v_dst, v_dst.clone(), v_ts, v_eid
            # )
            
            # v_logits = model.affinity_score(v_src_emb, v_dst_emb).view(-1)
            
            # val_loss += criterion(v_logits, v_y).item()
            # #----- affinity score-----
        

            v_prob = torch.sigmoid(v_logits).detach().cpu().numpy()
            y_buf.append(v_y.detach().cpu().numpy())
            p_buf.append(v_prob)

    val_loss = val_loss / val_batches  

    y_true = np.concatenate(y_buf) if len(y_buf) else np.array([])
    y_prob = np.concatenate(p_buf) if len(p_buf) else np.array([])

    if y_true.size > 0 and np.unique(y_true).size > 1:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
        print(f"[VAL] AUC={auc:.4f}  PR-AUC={ap:.4f}")
    else:
        print("[VAL] AUC/PR-AUC 계산 불가(단일 클래스)")

    print(f"[Ep {epoch+1}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss; counter = 0
        torch.save({
            "tgn": model.state_dict(),
            "clf": fraud_classifier.state_dict()
        }, "best_tgn_v22_2.pth")
    else:
        counter += 1
        if counter > 3:
            print("Early stopping.")
            break
 

from sklearn.metrics import confusion_matrix
# === 체크포인트 로드 ===
ckpt = torch.load("best_tgn_v22_1.pth", map_location=device)
model.load_state_dict(ckpt["tgn"]); model.to(device).eval()
fraud_classifier.load_state_dict(ckpt["clf"]); fraud_classifier.to(device).eval()

@torch.no_grad()
def infer_probs(idx_seq, src_all, dst_all, ts_all, eids_all, eX_scaled, y_all, batch_size=256):
    # 시간순 정렬
    order = torch.argsort(ts_all[idx_seq])
    seq = idx_seq[order]

    if hasattr(model, "memory") and hasattr(model.memory, "reset_state"):
        model.memory.reset_state()

    probs, labels = [], []
    for start in range(0, len(seq), batch_size):
        b_idx = seq[start:start+batch_size]
        b_src = src_all[b_idx]
        b_dst = dst_all[b_idx]
        b_ts  = ts_all[b_idx].detach().cpu().numpy().astype(np.float32)
        b_eid = eids_all[b_idx]
        b_edge_attr = eX_scaled[b_idx]

        dummy_neg = b_dst.clone()  # TGN 구현상 negative_nodes 필요
        src_emb, dst_emb, _ = model.compute_temporal_embeddings(
            b_src, b_dst, dummy_neg, b_ts, b_eid
        )

        edge_repr = torch.cat([src_emb, dst_emb, b_edge_attr], dim=-1)
        logits = fraud_classifier(edge_repr).view(-1)
        p = torch.sigmoid(logits)

        probs.append(p.detach().cpu())
        labels.append(y_all[b_idx].detach().cpu())

    probs  = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy().astype(int)
    return probs, labels

# === (A) Validation: 최적 임계값 찾기 ===
val_probs, val_labels = infer_probs(
    torch.arange(len(val_src), device=device),
    val_src, val_dst, val_ts, val_edge_ids, val_eX_scaled, val_y
)

prec_v, rec_v, th_v = precision_recall_curve(val_labels, val_probs)
f1_v = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)
best_i = f1_v.argmax()
best_threshold = th_v[best_i]
print(f"[VAL] Best F1={f1_v[best_i]:.4f} @ threshold={best_threshold:.4f}")

# === (B) Test 평가 ===
test_probs, test_labels = infer_probs(
    torch.arange(len(test_src), device=device),
    test_src, test_dst, test_ts, test_edge_ids, test_eX_scaled, test_y
)

# --- 0.5 기준 지표 ---
test_preds_05 = (test_probs >= 0.5).astype(int)
print("\n=== Classification report (threshold=0.5) ===")
print(classification_report(test_labels, test_preds_05, target_names=["normal(0)", "fraud(1)"]))

auc_roc = roc_auc_score(test_labels, test_probs)
print(f"ROC-AUC = {auc_roc:.4f}")

# --- Best-F1 기준 지표 ---
test_preds_best = (test_probs >= best_threshold).astype(int)
print(f"\n=== Classification report (threshold={best_threshold:.4f}, from VAL best-F1) ===")
print(classification_report(test_labels, test_preds_best, target_names=["normal(0)", "fraud(1)"], digits=4))

# === (C) 시각화 ===
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sk_auc 

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

plt.rcParams['font.family'] = 'AppleGothic'  

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

model_name = "TGN"
data_name = "test"

# 혼동 행렬
cm = confusion_matrix(test_labels, test_preds_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상(예측)', '사기(예측)'],
            yticklabels=['정상(실제)', '사기(실제)'])
plt.title(f'{model_name} 모델 혼동 행렬 ({data_name})\n(Optimal Threshold: {best_threshold:.2f})')
plt.ylabel('실제 레이블')
plt.xlabel('예측 레이블')
plt.savefig(f'./img/{model_name}_{data_name}_confusion_matrix.png')
plt.close()

# PR 곡선
precision, recall, _ = precision_recall_curve(test_labels, test_probs)
pr_auc = sk_auc(recall, precision)   
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.title(f'{model_name} 모델 정밀도-재현율 곡선 ({data_name})')
plt.xlabel('재현율 (Recall)')
plt.ylabel('정밀도 (Precision)')
plt.legend()
plt.grid(True)
plt.savefig(f'./img/{model_name}_{data_name}_pr_curve.png')
plt.close()

# ROC 곡선
fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = sk_auc(fpr, tpr)   
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.title(f'{model_name} 모델 ROC 곡선 ({data_name})')
plt.xlabel('거짓 양성률 (False Positive Rate)')
plt.ylabel('참 양성률 (True Positive Rate)')
plt.legend()
plt.grid(True)
plt.savefig(f'./img/{model_name}_{data_name}_roc_curve.png')
plt.close()

# 예측 결과(확률, 실제 레이블)를 NumPy 파일로 저장
np.savez_compressed(
    f'./results/{model_name}_{data_name}_results.npz',
    labels=test_labels,
    probs=test_probs
)
 
import pickle

with open("card_le.pkl", "wb") as f:
    pickle.dump(card_le_global, f)

with open("merch_le.pkl", "wb") as f:
    pickle.dump(merch_le_global, f)

with open("edge_scaler.pkl", "wb") as f:
    pickle.dump(edge_scaler, f)

import torch


# NeighborFinder 저장
torch.save(neighbor_finder, "neighbor_finder.pt")

