import numpy as np
from preprocessing import preprocess_image, preprocess_image_array, select_liver_roi
from segmentation import best_candidate_region
from feature_extraction import (
    extract_liver_features,
    extract_mass_features,
    empty_mass_features,
    features_dict_to_vector,
)


def process_bgr(resized_bgr, enhanced_gray):

    liver_bbox = select_liver_roi(enhanced_gray)
    lx, ly, lw, lh = liver_bbox
    liver_roi_gray = enhanced_gray[ly:ly + lh, lx:lx + lw]

    liver_feats, liver_area = extract_liver_features(enhanced_gray, liver_bbox)

    mass_bbox = None
    mass_contour_local = best_candidate_region(liver_roi_gray)
    if mass_contour_local is not None:
        mass_contour = mass_contour_local + np.array([[lx, ly]])
        mass_feats = extract_mass_features(
            enhanced_gray, mass_contour, liver_area, liver_feats["liver_mean_intensity"]
        )
        mass_bbox = mass_feats["bbox"]
    else:
        mass_feats = empty_mass_features()

    merged = {**liver_feats, **mass_feats}
    vector = features_dict_to_vector(merged)

    return {
        "features": merged,
        "feature_vector": vector,
        "liver_bbox": liver_bbox,
        "mass_bbox": mass_bbox,
    }


def process_image_path(image_path):
    resized_bgr, enhanced_gray, _scale = preprocess_image(image_path)
    result = process_bgr(resized_bgr, enhanced_gray)
    result["resized_bgr"] = resized_bgr
    return result
