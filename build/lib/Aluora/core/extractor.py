from Aluora.models.bart_cnn import BartCNN
import torch
import torch.nn as nn
from lettucedetect.models.inference import HallucinationDetector
from transformers import AutoModelForSequenceClassification, logging as hf_logging
from scipy.stats import entropy
import numpy as np
import torch.nn.functional as F
from Aluora.models.simple_densenet import DropoutDenseNet
import json
import os
from huggingface_hub import hf_hub_download

hf_logging.set_verbosity_error()

model = BartCNN()

def hallucination_metrics(context: str, question: str, answer: str, output_json_path=None):
    features_to_extract = {
        "mtp": True,
        "avgtp": True,
        "MDVTP": True,
        "MMDVP": True
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = model.extractFeatures(context, question, answer, features_to_extract)

    prob_class_1, predicted_class, mutual_info = halludetect_hallucination_risk(
        features_dict=features,
        input_dim=4,
        hidden_dim=512,
        device=device,
    )

    score_hhem = hhem_hallucination_metrics(context, question, answer).item()

    lettuce_spans_output = lettuce_hall_metrics(context, question, answer)

    results = {
        "halludetect": {
            "predicted_class": predicted_class,
            "label": "🤖 Hallucination" if predicted_class == 0 else "✅ No Hallucination",
            "probability_class_1": round(prob_class_1, 4),
            "mutual_information": round(mutual_info, 4)
        },
        "hhem": {
            "prob_no_hallucination": round(score_hhem, 4),
            "risk_level": get_risk_level(1 - score_hhem)
        },
        "lettuce": {
            "detected_spans": lettuce_spans_output,
            "estimated_risk": get_lettuce_risk_level(lettuce_spans_output)
        }
    }

    print_results(results)

    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    return results

def halludetect_hallucination_risk(features_dict: dict, input_dim: int,
                                    hidden_dim: int, output_dim: int = 2, dropout_prob: float = 0.3,
                                    device='cpu'):
    ordered_keys = sorted(features_dict.keys())
    feature_values = [float(features_dict[k]) for k in ordered_keys]
    input_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)

    if input_tensor.shape[1] != input_dim:
        raise ValueError(f"Input dimension mismatch: expected {input_dim}, got {input_tensor.shape[1]}")

    model_instance = DropoutDenseNet(input_dim, hidden_dim, output_dim, dropout_prob)
    model_path = hf_hub_download(
        repo_id="PedroooSaarm/HD-Dropout-Dense-Net",
        filename="model_dropout_densenet.pth"
    )
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.train()

    mean_preds, std_preds, mutual_info = mc_dropout_predict(model_instance, input_tensor.to(device), num_samples=100)
    prob_class_1 = mean_preds[0, 1]
    predicted_class = np.argmax(mean_preds[0])
    return prob_class_1, int(predicted_class), float(mutual_info)

def mc_dropout_predict(model, input_tensor, num_samples=100):
    model.train()
    preds = []

    with torch.no_grad():
        for _ in range(num_samples):
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)
            preds.append(prob.cpu().numpy())

    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)

    entropy_mean = entropy(mean_preds.T)
    entropy_samples = np.mean([entropy(p.T) for p in preds], axis=0)
    mutual_info = entropy_mean - entropy_samples

    return mean_preds, std_preds, mutual_info

def hhem_hallucination_metrics(context: str, question: str, answer: str):
    premise = context + " " + question
    hypothesis = answer
    hhem = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)
    pairs = [(premise, hypothesis)]
    score = hhem.predict(pairs)
    return score

def lettuce_hall_metrics(context: str, question: str, answer: str):
    detector = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    contexts = [context]
    predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
    return predictions

def get_risk_level(score):
    if score >= 0.75:
        return "🚨 HIGH"
    elif score >= 0.5:
        return "⚠️ MEDIUM"
    else:
        return "🟢 LOW"

def get_lettuce_risk_level(spans):
    if not spans:
        return "🟢 No hallucination"
    max_conf = max([f["confidence"] for f in spans])
    if max_conf >= 0.95:
        return "🚨 HIGH"
    elif max_conf >= 0.80:
        return "⚠️ MEDIUM"
    else:
        return "🟢 LOW"

def print_results(results):
    print("\n📊 \033[1mAluora Evaluation\033[0m\n")

    print("\033[1m🧠 HalluDetect:\033[0m")
    print(f"  → Predicted class: {results['halludetect']['predicted_class']} ({results['halludetect']['label']})")
    print(f"  → Probability class 1: {results['halludetect']['probability_class_1']}")
    print(
        f"  → Mutual information: {results['halludetect']['mutual_information']} (0 = confident, ~0.7 = very uncertain)\n")

    print("\033[1m📐 HHEM:\033[0m")
    print(f"  → Probability of no hallucination: {results['hhem']['prob_no_hallucination']}")
    print(f"  → Risk level: {results['hhem']['risk_level']}\n")

    print("\033[1m🥬 Lettuce Detect:\033[0m")
    for i, frag in enumerate(results['lettuce']['detected_spans'], 1):
        print(f"  → Hallucinated span {i}: '{frag['text']}' (confidence: {frag['confidence']:.4f})")
    print(f"  → Estimated risk: {results['lettuce']['estimated_risk']}\n")

    print("\033[1m✅ Evaluation complete.\033[0m\n")
