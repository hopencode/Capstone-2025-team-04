# src/main.py

import argparse
import pandas as pd
import torch
import joblib
import warnings

import src.config as config
from src.train import run_enhanced_fraud_detection_system
from src.evaluate import analyze_saved_results

warnings.filterwarnings('ignore'


def main():
    """메인 실행 함수"""

    # 1. 커맨드 라인 인자 파서 설정
    parser = argparse.ArgumentParser(description="Enhanced MLP Fraud Detection System")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'analyze'],
        help="실행 모드를 선택하세요: 'train' 또는 'analyze'"
    )
    args = parser.parse_args()

    # 2. 선택된 모드에 따라 작업 수행
    if args.mode == 'train':
        print("---< Train Mode >---")

        print("데이터 로딩 중...")
        train_meta_df = pd.read_csv(config.TRAIN_META_PATH)
        train_embeddings_df = pd.read_csv(config.TRAIN_EMBEDDINGS_PATH)
        val_meta_df = pd.read_csv(config.VAL_META_PATH)
        val_embeddings_df = pd.read_csv(config.VAL_EMBEDDINGS_PATH)
        test_meta_df = pd.read_csv(config.TEST_META_PATH)
        test_embeddings_df = pd.read_csv(config.TEST_EMBEDDINGS_PATH)
        users_df = pd.read_csv(config.USERS_PATH)
        cards_df = pd.read_csv(config.CARDS_PATH)

        enhanced_model, results = run_enhanced_fraud_detection_system(
            train_meta_df, train_embeddings_df,
            val_meta_df, val_embeddings_df,
            test_meta_df, test_embeddings_df,
            users_df, cards_df,
            use_focal_loss=config.USE_FOCAL_LOSS,
            use_smote=config.USE_SMOTE,
            use_optimal_threshold=config.USE_OPTIMAL_THRESHOLD
        )

        print("\n최종 모델 및 결과 저장 중...")
        torch.save(enhanced_model.state_dict(), config.FINAL_MODEL_SAVE_PATH)
        joblib.dump(results, config.RESULTS_PKL_PATH)
        print(f"파일 저장 완료: {config.FINAL_MODEL_SAVE_PATH}, {config.RESULTS_PKL_PATH}")

        print("\n---< Train Mode Complete >---")

    elif args.mode == 'analyze':
        print("---< Analyze Mode >---")

        # 저장된 결과 분석 및 시각화 (evaluate.py의 함수 호출)
        analyze_saved_results()

        print("\n---< Analyze Mode Complete >---")


if __name__ == '__main__':
    main()