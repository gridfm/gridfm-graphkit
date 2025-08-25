def compute_cm_metrics(y_test, y_pred, model_name, label_plot):
    """
    Compute confusion matrix (TP,FP,TN,FN) for predicted overleads along with their respective rates and accuracy metric.

    Parameters:
    - y_pred: predicted overlads
    - y_test: ground truth overloads
    - prediction_dir:
    - label_plot:
    """

    TP = (y_test & y_pred).sum()
    FP = ((~y_test) & y_pred).sum()
    TN = ((~y_test) & (~y_pred)).sum()
    FN = (y_test & (~y_pred)).sum()

    # accuracy
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print(f"Accuracy: {accuracy:.3f}")

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)
    # TODO change text to fit both overloadings and voltage violations
    print("Confusion Matrix:")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(
        f"GridFM\nTPR: {TPR:.3f} (percentage of overloadings correctly predicted)\nFPR: {FPR:.3f} (percentage of non-overloadings predicted as overloadings)\nTNR: {TNR:.2f}\nFNR: {FNR:.2f}",
    )
    with open(f"metrics_overloading_{model_name}.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n")
        f.write(f"{label_plot} Metrics:\n")
        f.write(f"TPR: {TPR:.5f} (percentage of overloadings correctly predicted)\n")
        f.write(
            f"FPR: {FPR:.5f} (percentage of non-overloadings predicted as overloadings)\n",
        )
        f.write(f"TNR: {TNR:.5f}\n")
        f.write(f"FNR: {FNR:.5f}\n")
    return TP, FP, TN, FN
