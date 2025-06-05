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

hf_logging.set_verbosity_error()

model = BartCNN()

def hallucination_metrics(context: str, question: str, answer: str, output_json_path=None):
    features_to_extract = {
        "mtp": True,
        "avgtp": True,
        "MDVTP": True,
        "MMDVP": True
    }
    model_path = "Aluora/weights/model_dropout_densenet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = model.extractFeatures(context, question, answer, features_to_extract)

    prob_class_1, predicted_class, mutual_info = halludetect_hallucination_risk(
        features_dict=features,
        model_path=model_path,
        input_dim=4,
        hidden_dim=512,
        device=device,
    )

    score_hhem = hhem_hallucination_metrics(context, question, answer).item()

    lettuce_spans_output = lettuce_hall_metrics(context, question, answer)

    resultados = {
        "halludetect": {
            "clase_predicha": predicted_class,
            "etiqueta": "🤖 Alucinación" if predicted_class == 0 else "✅ No Alucinación",
            "probabilidad_clase_1": round(prob_class_1, 4),
            "informacion_mutua": round(mutual_info, 4)
        },
        "hhem": {
            "prob_no_alucinacion": round(score_hhem, 4),
            "nivel_riesgo": calcular_nivel_riesgo(1 - score_hhem)
        },
        "lettuce": {
            "fragmentos_detectados": lettuce_spans_output,
            "riesgo_estimado": calcular_nivel_riesgo_lettuce(lettuce_spans_output)
        }
    }

    imprimir_resultados(resultados)

    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)

    return resultados

def halludetect_hallucination_risk(features_dict: dict, model_path: str, input_dim: int,
                                    hidden_dim: int, output_dim: int = 2, dropout_prob: float = 0.3,
                                    device='cpu'):
    ordered_keys = sorted(features_dict.keys())
    feature_values = [float(features_dict[k]) for k in ordered_keys]
    input_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)

    if input_tensor.shape[1] != input_dim:
        raise ValueError(f"Dimensión de entrada incorrecta: esperada {input_dim}, obtenida {input_tensor.shape[1]}")

    model_instance = DropoutDenseNet(input_dim, hidden_dim, output_dim, dropout_prob)
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

def calcular_nivel_riesgo(score):
    if score >= 0.75:
        return "🚨 ALTO"
    elif score >= 0.5:
        return "⚠️ MEDIO"
    else:
        return "🟢 BAJO"

def calcular_nivel_riesgo_lettuce(fragmentos):
    if not fragmentos:
        return "❓ DESCONOCIDO"
    max_conf = max([f["confidence"] for f in fragmentos])
    if max_conf >= 0.95:
        return "🚨 ALTO"
    elif max_conf >= 0.80:
        return "⚠️ MEDIO"
    else:
        return "🟢 BAJO"

def imprimir_resultados(resultados):
    print("\n📊 \033[1mEvaluación de Aluora\033[0m\n")

    print("\033[1m🧠 HalluDetect:\033[0m")
    print(f"  → Clase predicha: {resultados['halludetect']['clase_predicha']} ({resultados['halludetect']['etiqueta']})")
    print(f"  → Probabilidad clase 1: {resultados['halludetect']['probabilidad_clase_1']}")
    print(f"  → Información mutua: {resultados['halludetect']['informacion_mutua']}\n")

    print("\033[1m📐 HHEM:\033[0m")
    print(f"  → Probabilidad de no alucinación: {resultados['hhem']['prob_no_alucinacion']}")
    print(f"  → Nivel de riesgo: {resultados['hhem']['nivel_riesgo']}\n")

    print("\033[1m🥬 Lettuce Detect:\033[0m")
    for i, frag in enumerate(resultados['lettuce']['fragmentos_detectados'], 1):
        print(f"  → Fragmento alucinado{i}: '{frag['text']}' (confianza: {frag['confidence']:.4f})")
    print(f"  → Riesgo estimado: {resultados['lettuce']['riesgo_estimado']}\n")

    print("\033[1m✅ Evaluación completada.\033[0m\n")
