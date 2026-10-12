import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================
# 0) Paths and file lists
# =========================
ROOT_DIR = r"HF-EEG/ML-features"
OUT_XLSX = "RF_results.xlsx"

negative_files = ["negative01.npy", "negative02.npy", "negative03.npy", "negative04.npy"]
positive_files = ["positive01.npy", "positive02.npy", "positive03.npy", "positive04.npy"]
neutral_files  = ["neutral01.npy",  "neutral02.npy",  "neutral03.npy",  "neutral04.npy"]

LABEL_ORDER = [0, 1, 2]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


# =========================
# 1) Loading & concatenation
# =========================
def _load_npy_as_2d(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)

    if arr.size == 0:
        raise ValueError(f"Empty array: {path}")

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float64, copy=False)


def load_split(train_files_by_class, test_files_by_class, subject_dir: str):
    X_tr_list, y_tr_list = [], []
    X_te_list, y_te_list = [], []

    for cls_name, files in train_files_by_class.items():
        y_val = LABEL_MAP[cls_name]
        for fn in files:
            fp = os.path.join(subject_dir, fn)
            X = _load_npy_as_2d(fp)
            X_tr_list.append(X)
            y_tr_list.append(np.full((X.shape[0],), y_val, dtype=np.int64))

    for cls_name, files in test_files_by_class.items():
        y_val = LABEL_MAP[cls_name]
        for fn in files:
            fp = os.path.join(subject_dir, fn)
            X = _load_npy_as_2d(fp)
            X_te_list.append(X)
            y_te_list.append(np.full((X.shape[0],), y_val, dtype=np.int64))

    X_train = np.vstack(X_tr_list)
    y_train = np.concatenate(y_tr_list)
    X_test  = np.vstack(X_te_list)
    y_test  = np.concatenate(y_te_list)

    return X_train, y_train, X_test, y_test


def check_subject_files(subject_dir: str):
    all_files = negative_files + neutral_files + positive_files
    missing = [fn for fn in all_files if not os.path.exists(os.path.join(subject_dir, fn))]
    return missing


# =========================
# 2) Splits for a single subject
# =========================
def run_one_subject(subject_name: str, subject_dir: str, rf_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = check_subject_files(subject_dir)
    if missing:
        raise FileNotFoundError(
            f"[{subject_name}] 缺少文件（检查命名/路径）：\n" + "\n".join(missing)
        )

    rows = []
    split_id = 0
    total_splits = len(negative_files) * len(neutral_files) * len(positive_files)

    for neg_test in negative_files:
        for neu_test in neutral_files:
            for pos_test in positive_files:
                split_id += 1

                train_files = {
                    "negative": [f for f in negative_files if f != neg_test],
                    "neutral":  [f for f in neutral_files  if f != neu_test],
                    "positive": [f for f in positive_files if f != pos_test],
                }
                test_files = {
                    "negative": [neg_test],
                    "neutral":  [neu_test],
                    "positive": [pos_test],
                }

                X_train, y_train, X_test, y_test = load_split(train_files, test_files, subject_dir)

                clf = RandomForestClassifier(**rf_params)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average="macro")
                cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

                oob = float(getattr(clf, "oob_score_", np.nan))

                rows.append({
                    "subject": subject_name,
                    "split_id": split_id,
                    "total_splits": total_splits,
                    "neg_test": neg_test,
                    "neu_test": neu_test,
                    "pos_test": pos_test,
                    "train_size": int(len(y_train)),
                    "test_size": int(len(y_test)),
                    "accuracy": float(acc),
                    "macro_f1": float(macro_f1),
                    "oob_score": oob,

                    "cm_00": int(cm[0, 0]), "cm_01": int(cm[0, 1]), "cm_02": int(cm[0, 2]),
                    "cm_10": int(cm[1, 0]), "cm_11": int(cm[1, 1]), "cm_12": int(cm[1, 2]),
                    "cm_20": int(cm[2, 0]), "cm_21": int(cm[2, 1]), "cm_22": int(cm[2, 2]),
                })

    df = pd.DataFrame(rows)

    best_acc_idx = df["accuracy"].idxmax()
    worst_acc_idx = df["accuracy"].idxmin()
    best_f1_idx = df["macro_f1"].idxmax()
    worst_f1_idx = df["macro_f1"].idxmin()

    summary = pd.DataFrame([{
        "subject": subject_name,
        "splits": int(df.shape[0]),

        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy":  df["accuracy"].std(ddof=1),
        "mean_macro_f1": df["macro_f1"].mean(),
        "std_macro_f1":  df["macro_f1"].std(ddof=1),
        "mean_oob_score": df["oob_score"].mean(),

        "best_accuracy": df.loc[best_acc_idx, "accuracy"],
        "best_acc_split_id": int(df.loc[best_acc_idx, "split_id"]),
        "best_acc_triplet": f"{df.loc[best_acc_idx, 'neg_test']}|{df.loc[best_acc_idx, 'neu_test']}|{df.loc[best_acc_idx, 'pos_test']}",

        "worst_accuracy": df.loc[worst_acc_idx, "accuracy"],
        "worst_acc_split_id": int(df.loc[worst_acc_idx, "split_id"]),
        "worst_acc_triplet": f"{df.loc[worst_acc_idx, 'neg_test']}|{df.loc[worst_acc_idx, 'neu_test']}|{df.loc[worst_acc_idx, 'pos_test']}",

        "best_macro_f1": df.loc[best_f1_idx, "macro_f1"],
        "best_f1_split_id": int(df.loc[best_f1_idx, "split_id"]),
        "best_f1_triplet": f"{df.loc[best_f1_idx, 'neg_test']}|{df.loc[best_f1_idx, 'neu_test']}|{df.loc[best_f1_idx, 'pos_test']}",

        "worst_macro_f1": df.loc[worst_f1_idx, "macro_f1"],
        "worst_f1_split_id": int(df.loc[worst_f1_idx, "split_id"]),
        "worst_f1_triplet": f"{df.loc[worst_f1_idx, 'neg_test']}|{df.loc[worst_f1_idx, 'neu_test']}|{df.loc[worst_f1_idx, 'pos_test']}",
    }])

    return df, summary


# =========================
# 3) Main: iterate over all subjects
# =========================
def main():
    subject_names = sorted([
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ])
    if not subject_names:
        raise FileNotFoundError(f"在 ROOT_DIR={ROOT_DIR} 下没有找到任何被试文件夹")

    rf_params = dict(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=2026,
    )

    all_detail = []
    all_summary = []

    total = len(subject_names)
    for i, subject in enumerate(subject_names, start=1):
        subject_dir = os.path.join(ROOT_DIR, subject)
        print(f"\n[{i}/{total}] Running subject: {subject}")

        df_detail, df_summary = run_one_subject(subject, subject_dir, rf_params)
        all_detail.append(df_detail)
        all_summary.append(df_summary)

        print(
            f"  -> done: splits={df_summary.loc[0,'splits']}, "
            f"mean_acc={df_summary.loc[0,'mean_accuracy']:.4f}, "
            f"mean_macro_f1={df_summary.loc[0,'mean_macro_f1']:.4f}, "
            f"mean_oob={df_summary.loc[0,'mean_oob_score']:.4f}"
        )

    df_all = pd.concat(all_detail, ignore_index=True)
    df_sum = pd.concat(all_summary, ignore_index=True)

    overall = pd.DataFrame([{
        "subjects_count": int(df_sum.shape[0]),
        "overall_splits": int(df_all.shape[0]),
        "overall_mean_accuracy": df_all["accuracy"].mean(),
        "overall_mean_macro_f1": df_all["macro_f1"].mean(),
        "overall_mean_oob_score": df_all["oob_score"].mean(),
        "overall_best_accuracy": df_all["accuracy"].max(),
        "overall_best_macro_f1": df_all["macro_f1"].max(),
    }])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_all.to_excel(writer, index=False, sheet_name="all_results")
        df_sum.to_excel(writer, index=False, sheet_name="summary_by_subject")
        overall.to_excel(writer, index=False, sheet_name="overall")

    print(f"\nAll done. Saved to: {OUT_XLSX}")


if __name__ == "__main__":
    main()
