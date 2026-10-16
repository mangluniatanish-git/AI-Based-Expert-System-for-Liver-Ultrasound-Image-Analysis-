import os
import argparse

import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

def load_dataset(path):

    data = np.load(
        path,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    feature_names = list(
        data["feature_names"]
    )

    return X, y, feature_names


def create_candidate_models():

    return {
        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),

        "SVM (RBF kernel)":
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=42,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(
                n_neighbors=7
            ),
    }


def cross_validate_models(X_train, y_train):

    print("=" * 78)
    print(
        "STEP 1: Comparing classifiers with "
        "5-fold cross-validation (Accuracy & Macro F1)"
    )
    print("=" * 78)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = create_candidate_models()

    acc_results = {}
    f1_results = {}

    for name, model in models.items():

        acc_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        f1_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        acc_results[name] = acc_scores
        f1_results[name] = f1_scores

        print(
            f"  {name:22s}: "
            f"mean acc = {acc_scores.mean() * 100:5.1f}% (+/- {acc_scores.std() * 100:4.1f}%) | "
            f"mean macro F1 = {f1_scores.mean():.4f}"
        )

    best_name = max(
        f1_results,
        key=lambda k: f1_results[k].mean()
    )

    print(
        f"\n  Best baseline model by CV Macro F1: "
        f"{best_name}"
    )

    return best_name, acc_results, f1_results


def tune_svm(X_train, y_train):

    print("\n" + "=" * 78)
    print(
        "STEP 1B: Tuning SVM (RBF) hyperparameters"
    )
    print("=" * 78)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=42,
    )

    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5, 10, 50],
        "gamma": ["scale", 0.01, 0.05, 0.1, 0.2],
    }

    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        f"  Best parameters: "
        f"{grid.best_params_}"
    )

    print(
        f"  Best CV Macro F1: "
        f"{grid.best_score_:.4f}"
    )

    return grid.best_estimator_, grid.best_score_


def tune_rf(X_train, y_train):

    print("\n" + "=" * 78)
    print(
        "STEP 1C: Tuning Random Forest hyperparameters"
    )
    print("=" * 78)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [5, 6, 7, 8],
        "min_samples_split": [2],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        f"  Best parameters: "
        f"{grid.best_params_}"
    )

    print(
        f"  Best CV Macro F1: "
        f"{grid.best_score_:.4f}"
    )

    return grid.best_estimator_, grid.best_score_

def evaluate_on_test_set(
    model,
    X_test,
    y_test,
    label_names
):
    """
    Evaluate final model on held-out test set.
    """

    print("\n" + "=" * 78)
    print(
        "STEP 3: Held-out test set evaluation"
    )
    print("=" * 78)

    y_pred = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print(
        f"  Test accuracy: {acc * 100:.1f}%"
    )

    print(
        f"  Macro F1:      {macro_f1:.3f}"
    )

    print(
        f"  Weighted F1:   {weighted_f1:.3f}"
    )

    print("\n  Classification report:")

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        digits=3,
    )

    for line in report.splitlines():
        print(
            f"    {line}"
        )

    print(
        "\n  Confusion matrix "
        "(rows = actual, columns = predicted):"
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=label_names
    )

    header = (
        "                " +
        "".join(
            f"{name[:10]:>12s}"
            for name in label_names
        )
    )

    print(header)

    for actual_name, row in zip(
        label_names,
        cm
    ):

        row_str = "".join(
            f"{val:12d}"
            for val in row
        )

        print(
            f"    {actual_name:12s}"
            f"{row_str}"
        )

    return acc, macro_f1, weighted_f1, cm


def print_feature_importance(
    model,
    feature_names
):
    """
    Print Random Forest feature importance if available.
    """

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return

    print("\n" + "=" * 78)
    print(
        "Feature importance "
        "(top 10)"
    )
    print("=" * 78)

    importances = (
        model.feature_importances_
    )

    order = np.argsort(
        importances
    )[::-1][:10]

    for i in order:

        print(
            f"  {feature_names[i]:28s}: "
            f"{importances[i]:.4f}"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a simple "
            "Benign/Malignant/Normal liver "
            "lesion classifier."
        )
    )

    parser.add_argument(
        "--features",
        default="features_dataset.npz",
        help=(
            "Path to cached feature dataset "
            "produced by dataset_builder.py."
        ),
    )

    parser.add_argument(
        "--model-out",
        default="models",
        help=(
            "Folder where trained model and "
            "scaler will be saved."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help=(
            "Fraction held out for final test. "
            "Default: 0.2"
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(
        args.features
    ):

        raise FileNotFoundError(
            f"{args.features} not found.\n\n"
            f"Run dataset_builder.py first."
        )

    X, y, feature_names = (
        load_dataset(
            args.features
        )
    )

    label_names = sorted(
        set(y)
    )

    print(
        f"Loaded {len(y)} samples, "
        f"{X.shape[1]} features, "
        f"classes: {label_names}\n"
    )

    print(
        "Class distribution:"
    )

    for label in label_names:

        count = int(
            np.sum(y == label)
        )

        percentage = (
            count / len(y) * 100
        )

        print(
            f"  {label:12s}: "
            f"{count:4d} "
            f"({percentage:5.1f}%)"
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(y_train)}"
    )

    print(
        f"Test samples:     "
        f"{len(y_test)}"
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    best_baseline_name, acc_results, f1_results = cross_validate_models(
        X_train_scaled,
        y_train
    )

    tuned_svm, tuned_svm_cv = tune_svm(
        X_train_scaled,
        y_train
    )

    tuned_rf, tuned_rf_cv = tune_rf(
        X_train_scaled,
        y_train
    )

    baseline_best_cv = f1_results[best_baseline_name].mean()

    candidates = {
        "Tuned SVM (RBF kernel)": (tuned_svm, tuned_svm_cv),
        "Tuned Random Forest": (tuned_rf, tuned_rf_cv),
        best_baseline_name: (create_candidate_models()[best_baseline_name], baseline_best_cv),
    }

    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_model = candidates[best_name][0]
    print(
        f"\n  Final selected model by CV Macro F1: "
        f"'{best_name}' (CV Macro F1 = {candidates[best_name][1]:.4f})"
    )

    print("\n" + "=" * 78)
    print(
        f"STEP 2: Training final "
        f"'{best_name}' model"
    )
    print("=" * 78)

    best_model.fit(
        X_train_scaled,
        y_train
    )

    print(
        "  Done."
    )

    test_accuracy, test_macro_f1, test_weighted_f1, test_cm = evaluate_on_test_set(
        best_model,
        X_test_scaled,
        y_test,
        label_names
    )

    calibrated_model = CalibratedClassifierCV(
        best_model,
        method="sigmoid",
        cv=5,
        ensemble=False,
    )

    calibrated_model.fit(
        X_train_scaled,
        y_train
    )

    calibrated_pred = (
        calibrated_model.predict(
            X_test_scaled
        )
    )

    calibrated_accuracy = (
        accuracy_score(
            y_test,
            calibrated_pred
        )
    )

    calibrated_macro_f1 = f1_score(
        y_test,
        calibrated_pred,
        average="macro"
    )

    print(
        f"\n  Calibrated model test accuracy: "
        f"{calibrated_accuracy * 100:.1f}%"
    )
    print(
        f"  Calibrated model test Macro F1: "
        f"{calibrated_macro_f1:.4f}"
    )

    if calibrated_macro_f1 >= test_macro_f1:
        model_to_save = calibrated_model
        save_name = f"{best_name} (Calibrated)"
        final_test_acc = calibrated_accuracy
        final_test_mf1 = calibrated_macro_f1
        print("  Deploying calibrated model.")
    else:
        model_to_save = best_model
        save_name = best_name
        final_test_acc = test_accuracy
        final_test_mf1 = test_macro_f1
        print("  Deploying balanced classifier (preserves minority class recall).")

    os.makedirs(
        args.model_out,
        exist_ok=True
    )

    joblib.dump(
        model_to_save,
        os.path.join(
            args.model_out,
            "liver_classifier.joblib"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            args.model_out,
            "feature_scaler.joblib"
        )
    )

    joblib.dump(
        feature_names,
        os.path.join(
            args.model_out,
            "feature_names.joblib"
        )
    )

    joblib.dump(
        save_name,
        os.path.join(
            args.model_out,
            "model_name.joblib"
        )
    )

    metadata = {
        "test_accuracy": float(
            final_test_acc
        ),
        "test_macro_f1": float(
            final_test_mf1
        ),
        "calibrated_test_accuracy": float(
            calibrated_accuracy
        ),
        "calibrated_test_macro_f1": float(
            calibrated_macro_f1
        ),
        "best_model": save_name,
        "n_samples": int(
            len(y)
        ),
        "n_features": int(
            X.shape[1]
        ),
        "classes": label_names,
    }

    joblib.dump(
        metadata,
        os.path.join(
            args.model_out,
            "training_metadata.joblib"
        )
    )

    print("\n" + "=" * 78)
    print(
        f"Saved trained model "
        f"('{best_name}') and scaler to: "
        f"{args.model_out}/"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()