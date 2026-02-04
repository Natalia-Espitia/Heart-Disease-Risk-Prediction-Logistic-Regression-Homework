import numpy as np
import json
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    w = np.load(os.path.join(model_dir, "weights.npy"))
    b = np.load(os.path.join(model_dir, "bias.npy")).item()
    return {"w": w, "b": b}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["inputs"], dtype=float).reshape(1, -1)
    else:
        raise ValueError("Unsupported content type: " + str(request_content_type))

def predict_fn(input_data, model):
    w, b = model["w"], model["b"]
    prob = sigmoid(input_data @ w + b)
    return {"heart_disease_probability": float(prob[0][0])}

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction), response_content_type
    else:
        raise ValueError("Unsupported response content type: " + str(response_content_type))
