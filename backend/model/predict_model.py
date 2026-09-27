"""
Load olive_classifier.keras and predict one leaf image.

Usage:
    python predict_model.py --image path/to/leaf.jpg
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTIONS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "disease_descriptions.json")

CONTAGION_RADIUS_M = {
    "peacock_spot": 12,
    "olive_knot": 8,
    "aculus_olearius": 10,
    "verticillium_wilt": 15,
    "anthracnose": 10,
    "diseased": 12,
    "healthy": 0,
}


def load_descriptions():
    with open(DESCRIPTIONS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("_readme", None)
    return data


def predict(image_path, model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, "olive_classifier.keras"))
    with open(os.path.join(model_dir, "class_indices.json"), "r", encoding="utf-8") as handle:
        class_indices = json.load(handle)

    image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    scores = model.predict(array, verbose=0)[0]
    top_idx = int(np.argmax(scores))
    confidence = float(scores[top_idx])
    class_key = class_indices[str(top_idx)]

    descriptions = load_descriptions()
    desc = descriptions.get(class_key, {})
    diseased = class_key != "healthy"
    severity = "low" if not diseased else "moderate"

    return {
        "detected": diseased,
        "disease_id": class_key,
        "disease_name_en": desc.get("disease_name_en", class_key),
        "disease_name_fr": desc.get("disease_name_fr", ""),
        "disease_name_ar_derja": desc.get("disease_name_ar_derja", ""),
        "pathogen_type": desc.get("pathogen_type", "unknown"),
        "confidence": round(confidence, 3),
        "severity": severity,
        "visual_evidence": (
            f"Local classifier prediction (MobileNetV2), class '{class_key}' "
            f"at {confidence:.1%} confidence."
        ),
        "contagion_radius_meters": CONTAGION_RADIUS_M.get(class_key, 0),
        "treatment_direction": desc.get("typical_visual_description", ""),
        "requires_expert_confirmation": diseased,
        "voice_script": desc.get("voice_script", ""),
        "voice_script_key": desc.get("voice_script_key", "uncertain_retry"),
        "voice_script_slots": {
            "disease_name": desc.get("disease_name_en", class_key),
            "severity": severity,
            "action": desc.get("typical_visual_description", ""),
        },
        "source": "local_classifier",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_dir", default=SCRIPT_DIR)
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
