import os
import argparse

import cv2
import joblib
import numpy as np

from pipeline import process_image_path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

LABEL_COLORS = {
    "Malignant": (0, 0, 255),   
    "Benign": (255, 0, 0),      
    "Normal": (0, 200, 0),      
}


def load_model(model_dir):
    model = joblib.load(os.path.join(model_dir, "liver_classifier.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "feature_scaler.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    model_name = joblib.load(os.path.join(model_dir, "model_name.joblib"))
    return model, scaler, feature_names, model_name


def predict_and_probabilities(model, vec_scaled):
    clf = getattr(model, "estimator", model)
    prediction = str(clf.predict(vec_scaled)[0])

    probabilities = {}
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(vec_scaled)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        probabilities = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
        max_cls = max(probabilities, key=probabilities.get)
        if max_cls != prediction:
            probabilities[prediction], probabilities[max_cls] = (
                probabilities[max_cls],
                probabilities[prediction],
            )
    elif hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(vec_scaled)[0]
        probabilities = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec_scaled)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, probs)}

    return prediction, probabilities


def analyze_image(image_path, model, scaler, feature_names):
    pipeline_result = process_image_path(image_path)

    vec = pipeline_result["feature_vector"].reshape(1, -1)
    vec_scaled = scaler.transform(vec)

    prediction, probabilities = predict_and_probabilities(model, vec_scaled)

    result = {
        "image_path": image_path,
        "liver_bbox": pipeline_result["liver_bbox"],
        "mass_bbox": pipeline_result["mass_bbox"],
        "features": pipeline_result["features"],
        "prediction": prediction,
        "probabilities": probabilities,
    }
    return result, pipeline_result["resized_bgr"]


def annotate_result(resized_bgr, result):
    annotated = resized_bgr.copy()
    lx, ly, lw, lh = result["liver_bbox"]
    cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (200, 200, 200), 1)

    prediction = result["prediction"]
    confidence = result["probabilities"].get(prediction, 0.0)
    conf_str = f"  {confidence * 100:.1f}%" if confidence > 0 else ""

    if result["mass_bbox"] is not None and prediction != "Normal":
        mx, my, mw, mh = result["mass_bbox"]
        color = LABEL_COLORS.get(prediction, (255, 255, 255))
        cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), color, 2)
        label = f"{prediction}{conf_str}"
        text_y = my - 8 if my - 8 > 10 else my + mh + 18
        cv2.putText(annotated, label, (mx, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        color = LABEL_COLORS.get(prediction, (255, 255, 255))
        cv2.putText(annotated, f"{prediction}{conf_str}", (lx + 5, ly + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return annotated


def print_report(result):
    print("=" * 78)
    print("LIVER ULTRASOUND CLASSIFICATION RESULT")
    print("=" * 78)
    print(f"Image      : {result['image_path']}")
    print(f"Liver ROI  : {result['liver_bbox']}")
    print(f"Lesion ROI : {result['mass_bbox'] if result['mass_bbox'] else 'no candidate region detected'}")
    print()
    print(f"PREDICTED CLASS: {result['prediction']}")
    if result["probabilities"]:
        print("\nClass probabilities:")
        for cls, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
            bar = "#" * int(p * 40)
            print(f"  {cls:12s} {p*100:5.1f}%  {bar}")
    print()
    print("NOTE: This is an automated screening output, not a medical")
    print("diagnosis. Final interpretation must be made by a qualified")
    print("medical professional.")
    print("=" * 78)


def process_one(image_path, model, scaler, feature_names, output_dir):
    try:
        result, resized_bgr = analyze_image(image_path, model, scaler, feature_names)
        print_report(result)

        annotated = annotate_result(resized_bgr, result)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_predicted.png")
        cv2.imwrite(out_path, annotated)
        print(f"\nAnnotated image saved to: {out_path}\n")
        return True
    except Exception as e:
        print(f"\n[FAILED] {image_path}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Predict Benign / Malignant / Normal on a new liver "
                    "ultrasound image using the trained model."
    )
    parser.add_argument("--input", required=True,
                         help="Path to a single image, or a folder of images.")
    parser.add_argument("--output", default="predictions",
                         help="Folder to save annotated image(s) into. "
                              "Default: ./predictions")
    parser.add_argument("--model-dir", default="models",
                         help="Folder containing the trained model files "
                              "(from train_model.py). Default: ./models")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "liver_classifier.joblib")):
        raise FileNotFoundError(
            f"No trained model found in {args.model_dir}/. Run train_model.py first."
        )

    model, scaler, feature_names, model_name = load_model(args.model_dir)
    print(f"Loaded trained model: {model_name}\n")

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        image_paths = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not image_paths:
            print(f"No images found in folder: {args.input}")
            return
        print(f"Found {len(image_paths)} image(s) in {args.input}\n")
        ok = sum(process_one(p, model, scaler, feature_names, args.output) for p in image_paths)
        print("=" * 78)
        print(f"Batch complete: {ok}/{len(image_paths)} image(s) processed successfully.")
        print("=" * 78)

    elif os.path.isfile(args.input):
        process_one(args.input, model, scaler, feature_names, args.output)

    else:
        print(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()
