import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                               roc_curve, auc, precision_recall_curve,
                               precision_score, recall_score, f1_score)
from datetime import datetime
import os

def analyze_saved_results():
    """
    저장된 enhanced_fraud_results.pkl 파일을 로드하여 완전한 분석 수행
    """
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                                 roc_curve, auc, precision_recall_curve,
                                 precision_score, recall_score, f1_score)
    from datetime import datetime
    import os

    # 저장된 결과 로드
    print("저장된 결과 로드 중...")
    results = joblib.load('enhanced_fraud_results.pkl')

    # 저장된 테스트 데이터 확인
    if 'y_test' not in results:
        print("ERROR: 저장된 파일에 테스트 데이터가 없습니다.")
        print("run_enhanced_fraud_detection_system 함수를 수정하고 다시 실행해주세요.")
        return None

    # 올바른 테스트 데이터 사용
    y_test = results['y_test']
    test_preds = results['predictions']['enhanced']
    ensemble_preds = results['predictions']['ensemble']

    print(f"\n데이터 크기 확인:")
    print(f"y_test: {len(y_test)}")
    print(f"enhanced predictions: {len(test_preds)}")
    print(f"ensemble predictions: {len(ensemble_preds)}")

    # Figure 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. ROC Curves
    ax = axes[0, 0]
    fpr_enhanced, tpr_enhanced, _ = roc_curve(y_test, test_preds)
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_preds)

    roc_auc_enhanced = auc(fpr_enhanced, tpr_enhanced)
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

    # ML 모델들의 ROC (있는 경우)
    if 'ml_results' in results:
        colors = ['green', 'orange', 'purple']
        for idx, (model_name, model_result) in enumerate(results['ml_results'].items()):
            model_preds = model_result['predictions']
            fpr, tpr, _ = roc_curve(y_test, model_preds)
            roc_auc_model = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx], alpha=0.7, linewidth=2,
                    label=f'{model_name} (AUC = {roc_auc_model:.4f})')

    ax.plot(fpr_enhanced, tpr_enhanced, 'b-', linewidth=2.5,
            label=f'Enhanced NN (AUC = {roc_auc_enhanced:.4f})')
    ax.plot(fpr_ensemble, tpr_ensemble, 'r-', linewidth=3,
            label=f'Final Ensemble (AUC = {roc_auc_ensemble:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves - All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    precision_enhanced, recall_enhanced, _ = precision_recall_curve(y_test, test_preds)
    precision_ensemble, recall_ensemble, _ = precision_recall_curve(y_test, ensemble_preds)

    ax.plot(recall_enhanced, precision_enhanced, 'b-', linewidth=2.5, label='Enhanced NN')
    ax.plot(recall_ensemble, precision_ensemble, 'r-', linewidth=3, label='Final Ensemble')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Score Distribution
    ax = axes[0, 2]
    normal_scores = ensemble_preds[y_test == 0]
    fraud_scores = ensemble_preds[y_test == 1]

    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', density=True)
    ax.hist(fraud_scores, bins=50, alpha=0.6, label='Fraud', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.set_xlabel('Prediction Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Score Distribution by Class', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Confusion Matrix - Enhanced NN
    ax = axes[1, 0]
    enhanced_binary = (test_preds > 0.5).astype(int)
    cm_enhanced = confusion_matrix(y_test, enhanced_binary)

    sns.heatmap(cm_enhanced, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'], ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix - Enhanced NN', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)

    # 5. Confusion Matrix - Final Ensemble
    ax = axes[1, 1]
    ensemble_binary = (ensemble_preds > 0.5).astype(int)
    cm_ensemble = confusion_matrix(y_test, ensemble_binary)

    sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'], ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix - Final Ensemble', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)

    # 6. Performance Metrics Bar Chart
    ax = axes[1, 2]

    # 성능 지표 계산
    metrics = {
        'Enhanced NN': {
            'Precision (Normal)': precision_score(y_test, enhanced_binary, pos_label=0),
            'Recall (Normal)': recall_score(y_test, enhanced_binary, pos_label=0),
            'F1 (Normal)': f1_score(y_test, enhanced_binary, pos_label=0),
            'Precision (Fraud)': precision_score(y_test, enhanced_binary, pos_label=1),
            'Recall (Fraud)': recall_score(y_test, enhanced_binary, pos_label=1),
            'F1 (Fraud)': f1_score(y_test, enhanced_binary, pos_label=1)
        },
        'Final Ensemble': {
            'Precision (Normal)': precision_score(y_test, ensemble_binary, pos_label=0),
            'Recall (Normal)': recall_score(y_test, ensemble_binary, pos_label=0),
            'F1 (Normal)': f1_score(y_test, ensemble_binary, pos_label=0),
            'Precision (Fraud)': precision_score(y_test, ensemble_binary, pos_label=1),
            'Recall (Fraud)': recall_score(y_test, ensemble_binary, pos_label=1),
            'F1 (Fraud)': f1_score(y_test, ensemble_binary, pos_label=1)
        }
    }

    # 막대 그래프
    x_labels = ['Prec\n(Norm)', 'Rec\n(Norm)', 'F1\n(Norm)',
                'Prec\n(Fraud)', 'Rec\n(Fraud)', 'F1\n(Fraud)']
    x = np.arange(len(x_labels))
    width = 0.35

    enhanced_scores = list(metrics['Enhanced NN'].values())
    ensemble_scores = list(metrics['Final Ensemble'].values())

    bars1 = ax.bar(x - width / 2, enhanced_scores, width, label='Enhanced NN', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, ensemble_scores, width, label='Final Ensemble', color='coral', alpha=0.8)

    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Performance Metrics by Class', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim([0.85, 1.02])
    ax.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Fraud Detection Model Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 이미지 저장
    save_dir = 'fraud_detection_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{save_dir}/fraud_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n그래프 저장 완료: {filename}")

    plt.show()

    # 텍스트 결과 출력
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE METRICS")
    print("=" * 80)

    print("\n1. Classification Report - Enhanced Neural Network:")
    print("-" * 50)
    print(classification_report(y_test, enhanced_binary,
                                target_names=['Normal', 'Fraud'], digits=4))

    print("\n2. Classification Report - Final Ensemble:")
    print("-" * 50)
    print(classification_report(y_test, ensemble_binary,
                                target_names=['Normal', 'Fraud'], digits=4))

    print("\n3. Detailed Metrics Comparison:")
    print("-" * 50)
    print(f"{'Model':<20} {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)

    for model_name in ['Enhanced NN', 'Final Ensemble']:
        for class_name in ['Normal', 'Fraud']:
            prec = metrics[model_name][f'Precision ({class_name})']
            rec = metrics[model_name][f'Recall ({class_name})']
            f1 = metrics[model_name][f'F1 ({class_name})']
            print(f"{model_name:<20} {class_name:<10} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
        print()

    print("4. AUC Scores:")
    print("-" * 50)
    print(f"Enhanced NN AUC: {roc_auc_enhanced:.4f}")
    print(f"Final Ensemble AUC: {roc_auc_ensemble:.4f}")
    if 'test_auc' in results:
        print(f"Stored Test AUC: {results['test_auc']:.4f}")
    if 'final_auc' in results:
        print(f"Stored Final AUC: {results['final_auc']:.4f}")

    print("\n5. Dataset Statistics:")
    print("-" * 50)
    print(f"Total test samples: {len(y_test):,}")
    print(f"Normal transactions: {np.sum(y_test == 0):,} ({np.sum(y_test == 0) / len(y_test) * 100:.2f}%)")
    print(f"Fraud transactions: {np.sum(y_test == 1):,} ({np.sum(y_test == 1) / len(y_test) * 100:.2f}%)")

    print("\n6. Error Analysis:")
    print("-" * 50)
    fp_enhanced = cm_enhanced[0, 1]
    fn_enhanced = cm_enhanced[1, 0]
    tp_enhanced = cm_enhanced[1, 1]
    tn_enhanced = cm_enhanced[0, 0]

    fp_ensemble = cm_ensemble[0, 1]
    fn_ensemble = cm_ensemble[1, 0]
    tp_ensemble = cm_ensemble[1, 1]
    tn_ensemble = cm_ensemble[0, 0]

    print("Enhanced NN:")
    print(f"  - True Positives:  {tp_enhanced:,} (correctly detected fraud)")
    print(f"  - True Negatives:  {tn_enhanced:,} (correctly identified normal)")
    print(f"  - False Positives: {fp_enhanced:,} ({fp_enhanced / np.sum(y_test == 0) * 100:.2f}% of normal)")
    print(f"  - False Negatives: {fn_enhanced:,} ({fn_enhanced / np.sum(y_test == 1) * 100:.2f}% of fraud)")
    print(f"  - Detection Rate:  {tp_enhanced / np.sum(y_test == 1) * 100:.2f}%")
    print(f"  - Accuracy:        {(tp_enhanced + tn_enhanced) / len(y_test) * 100:.2f}%")

    print("\nFinal Ensemble:")
    print(f"  - True Positives:  {tp_ensemble:,} (correctly detected fraud)")
    print(f"  - True Negatives:  {tn_ensemble:,} (correctly identified normal)")
    print(f"  - False Positives: {fp_ensemble:,} ({fp_ensemble / np.sum(y_test == 0) * 100:.2f}% of normal)")
    print(f"  - False Negatives: {fn_ensemble:,} ({fn_ensemble / np.sum(y_test == 1) * 100:.2f}% of fraud)")
    print(f"  - Detection Rate:  {tp_ensemble / np.sum(y_test == 1) * 100:.2f}%")
    print(f"  - Accuracy:        {(tp_ensemble + tn_ensemble) / len(y_test) * 100:.2f}%")

    return metrics