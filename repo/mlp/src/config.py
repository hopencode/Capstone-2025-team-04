# src/config.py

from pathlib import Path

# 1. 경로 설정
BASE_DIR = Path(__file__).parent.parent  # 프로젝트 최상위 디렉토리 (MLP/)
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "trained_model"
RESULTS_DIR = BASE_DIR / "results"

# 입력 데이터 파일 경로
TRAIN_META_PATH = DATA_DIR / "transformer_train_window_meta_80_V22.csv"
TRAIN_EMBEDDINGS_PATH = DATA_DIR / "transformer_train_embeddings_80_V22.csv"
VAL_META_PATH = DATA_DIR / "transformer_train_window_meta_20_V22.csv"
VAL_EMBEDDINGS_PATH = DATA_DIR / "transformer_train_embeddings_20_V22.csv"
TEST_META_PATH = DATA_DIR / "transformer_test_window_meta_V22.csv"
TEST_EMBEDDINGS_PATH = DATA_DIR / "transformer_test_embeddings_V22.csv"
USERS_PATH = DATA_DIR / "users.csv"
CARDS_PATH = DATA_DIR / "cards.csv"

# 출력 파일/디렉토리 경로
BEST_MODEL_SAVE_PATH = MODEL_DIR / "best_enhanced_model.pth"
FINAL_MODEL_SAVE_PATH = MODEL_DIR / "final_enhanced_fraud_model.pth"
RESULTS_PKL_PATH = RESULTS_DIR / "enhanced_fraud_results.pkl"
ANALYSIS_IMG_DIR = RESULTS_DIR / "fraud_detection_results" # evaluate.py에서 사용

# 2. 일반 설정
SEED = 42
DEVICE = "cuda" # 또는 "cpu"

# 3. 학습 제어 파라미터
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 20  # Early stopping을 위한 patience
USE_FOCAL_LOSS = True
USE_SMOTE = False
USE_OPTIMAL_THRESHOLD = True
SMOTE_STRATEGY = 0.3 # SMOTE 샘플링 비율

# 4. 모델 구조 파라미터 (EnhancedCustomerAwareEnsemble)
MODEL_PARAMS = {
    "ensemble_dim": 6,
    "hidden_dim": 256
}

# 5. 서브 모델 및 앙상블 파라미터
RF_PARAMS = {"n_estimators": 50, "max_depth": 5, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
LR_PARAMS = {"max_iter": 1000}

FINAL_ENSEMBLE_WEIGHTS = [0.4, 0.25, 0.25, 0.1] # Enhanced, RF, XGB, LR 순서