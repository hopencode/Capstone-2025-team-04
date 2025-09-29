import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class TGNEventStream:
    src: np.ndarray         
    dst: np.ndarray         
    t: np.ndarray           
    edge_attr: np.ndarray   
    y: np.ndarray           
    event_index: np.ndarray 
    info: Dict[str, Any]    

def build_tgn_event_stream_from_window_embeddings(
    emb_path: str,
    meta_path: str,
    time_col: str = "window_end_date",
    card_le: Optional[LabelEncoder] = None,
    merch_le: Optional[LabelEncoder] = None,
    include_zip_city: bool = True,
) -> Tuple[TGNEventStream, LabelEncoder, LabelEncoder]:

    # 1) 로드 & 병합
    emb  = pd.read_csv(emb_path)      
    meta = pd.read_csv(meta_path)     
    meta = meta.reset_index().rename(columns={"index": "index"})

    meta["index"] = emb["index"].values  
    df = pd.merge(meta, emb, on="index", how="inner")

    # 임베딩 컬럼
    emb_cols = [c for c in emb.columns if c != "index"]
    emb_dim  = len(emb_cols)

    # 2) 전처리
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' not found in meta.")
    df["card_id"] = df["card_id"].astype(str)
    df["last_merchant_id"] = df["last_merchant_id"].astype(str).fillna("UNKNOWN")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # zip/city 부호화 (이벤트 레벨)
    extra_feats = []
    if include_zip_city:
        # zip prefix (앞 3자리)
        if "last_zip" in df.columns:
            zip3 = df["last_zip"].astype(str).str.extract(r"(\d{3})")[0].fillna("0")
            df["_zip_prefix"] = pd.to_numeric(zip3, errors="coerce").fillna(0).astype(int)
        else:
            df["_zip_prefix"] = 0

        # city code (전기간 공통 라벨 인코딩)
        if "last_merchant_city" in df.columns:
            city_le = LabelEncoder().fit(df["last_merchant_city"].astype(str))
            df["_city_code"] = city_le.transform(df["last_merchant_city"].astype(str))
        else:
            df["_city_code"] = 0

        extra_feats = ["_zip_prefix", "_city_code"]

    # 3) 인코더 
    if card_le is None:
        card_le = LabelEncoder().fit(df["card_id"].astype(str))
    if merch_le is None:
        merch_le = LabelEncoder().fit(df["last_merchant_id"].astype(str))

    # 인코더 밖 ID 제거
    df = df[
        df["card_id"].isin(card_le.classes_) &
        df["last_merchant_id"].isin(merch_le.classes_)
    ].copy()

    num_cards  = len(card_le.classes_)
    num_merchs = len(merch_le.classes_)
    num_nodes  = num_cards + num_merchs

    # 4) 시간 정렬 (동시간 tie-break: index)
    df = df.sort_values([time_col, "index"]).reset_index(drop=True)

    # 5) 이벤트 배열
    src = card_le.transform(df["card_id"]).astype(np.int64)             
    dst = merch_le.transform(df["last_merchant_id"]).astype(np.int64)   
    dst = (dst + num_cards).astype(np.int64)                            
    t   = (df[time_col].astype("int64").values / 1e9).astype(np.float64) # ns -> sec

    if include_zip_city and len(extra_feats) > 0:
        edge_attr = np.hstack([
            df[emb_cols].values.astype(np.float32),
            df[extra_feats].values.astype(np.float32),
        ]).astype(np.float32)   # [E, emb_dim+2]
        edge_feat_dim = emb_dim + 2
    else:
        edge_attr = df[emb_cols].values.astype(np.float32)              
        edge_feat_dim = emb_dim

    if "last_fraud" in df.columns:
        y = pd.to_numeric(df["last_fraud"], errors="coerce").fillna(0).astype(np.float32).values
    else:
        y = np.zeros(len(df), dtype=np.float32)

    events = TGNEventStream(
        src=src,
        dst=dst,
        t=t,
        edge_attr=edge_attr,
        y=y,
        event_index=df["index"].values.astype(np.int64),
        info=dict(
            num_nodes=num_nodes,
            num_cards=num_cards,
            num_merchants=num_merchs,
            edge_feat_dim=edge_feat_dim,
            emb_dim=emb_dim,
            include_zip_city=include_zip_city,
            time_col=time_col,
            min_time=str(df[time_col].min()),
            max_time=str(df[time_col].max()),
        )
    )
    return events, card_le, merch_le
