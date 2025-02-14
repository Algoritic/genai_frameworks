import requests
import json
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from bert_score import score  
from textstat import flesch_reading_ease
from transformers import pipeline
from detoxify import Detoxify

nli_model = pipeline("text-classification", model="roberta-large-mnli")

# Load models
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define Model Server URL
MODEL_SERVER_URL = "http://localhost:8080/v1/chat/completions"

def compute_perplexity(response):
    """Compute perplexity using GPT-2"""
    """Lowe perplexity means the text is more predictable and fluent"""
    """Assesss if the model generates natural-sounding responses"""
    inputs = tokenizer(response, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        return round(torch.exp(loss).item(), 2)

def compute_heuristic_confidence(response):
    """Compute simple heuristic"""
    """Longer responses means higher confidence"""
    return round(min(len(response) / 50, 1.0), 2)

def compute_relevance(question, response):
    """Compute relevance using cosine similarity between question and response embeddings"""
    """Higher similarity means response is more on topic"""
    """Assess in detecting off-topic responses"""
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    return round(util.pytorch_cos_sim(question_embedding, response_embedding).item(), 2)

def compute_coherence(response):
    """Compute coherence using BERTScore"""
    """Compare sentence-to-sentence coherence to ensure the respnse flows logically"""
    sentences = response.split(". ")
    if len(sentences) < 2:
        return 1.00  
    _, _, coherence_score = score(sentences[:-1], sentences[1:], lang="en", rescale_with_baseline=True)
    return round(coherence_score.mean().item(), 2)

def compute_conciseness(response):
    words = response.split()
    avg_sentence_length = len(words) / (response.count('.') + 1)
    return round(max(0, 1 - (avg_sentence_length / 30)), 2)

def compute_readability(response):
    score = flesch_reading_ease(response)
    return round(min(max(score / 100, 0), 1), 2)

def compute_consistency(response):
    sentences = response.split(". ")
    for i in range(len(sentences) - 1):
        prediction = nli_model(f"{sentences[i]} {sentences[i+1]}")[0]
        if prediction['label'] == "contradiction":
            return 0.0  
    return 1.0

def compute_bias(response):
    toxicity_score = Detoxify('original').predict(response)['toxicity']
    return round(1 - toxicity_score, 2)

def compute_safety(response):
    safety_keywords = ["violence", "hate speech", "self-harm", "illegal"]
    return round(1 - any(word in response.lower() for word in safety_keywords), 2)

def compute_completeness(response, key_phrases):
    response_tokens = set(response.lower().split())
    matched_tokens = [phrase for phrase in key_phrases if phrase in response_tokens]
    return round(len(matched_tokens) / len(key_phrases), 2)

def compute_creativity(response, generic_responses):
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    generic_embeddings = [embedding_model.encode(g, convert_to_tensor=True) for g in generic_responses]
    max_similarity = max(util.pytorch_cos_sim(response_embedding, g_emb).item() for g_emb in generic_embeddings)
    return round(1 - max_similarity, 2)

def generate_and_evaluate(context, question):
    request_payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ]
    }
    start_time = time.time()
    response = requests.post(MODEL_SERVER_URL, json=request_payload, timeout=60)
    latency = round((time.time() - start_time), 2)

    if response.status_code == 200:
        response_data = response.json()
        generated_text = response_data["choices"][0]["message"]["content"]
        token_usage = response_data.get("usage", {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        })
        perplexity = compute_perplexity(generated_text)
        confidence = compute_heuristic_confidence(generated_text)
        relevance = compute_relevance(question, generated_text)
        coherence = compute_coherence(generated_text)
        conciseness = compute_conciseness(generated_text)
        readability = compute_readability(generated_text)
        consistency = compute_consistency(generated_text)
        bias = compute_bias(generated_text)
        safety = compute_safety(generated_text)

        return {
            "user_question": question,
            "context": context,
            "llm_response": generated_text,
            "usage": token_usage,
            "evaluation_metrics": {
                "quantitative_metrics": {
                    "perplexity": perplexity,
                    "confidence": confidence,
                    "relevance": relevance,
                    "latency": latency
                },
                "qualitative_metrics": {
                    "coherence": coherence,
                    "conciseness": conciseness,
                    "readability": readability,
                    "consistency": consistency,
                    "bias": bias,
                    "safety": safety
                }
            }
        }
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def evaluate_comparison(base_eval, specific_eval):
    """Compares evaluation metrics and determines Pass/Fail status with reason."""
    comparison = {}
    pass_count = 0
    fail_count = 0
    improvements = []
    degradations = []
    unchanged = []

    base_metrics = base_eval.get("evaluation_metrics", {})
    specific_metrics = specific_eval.get("evaluation_metrics", {})

    for category in ["quantitative_metrics", "qualitative_metrics"]:
        base_category = base_metrics.get(category, {})
        specific_category = specific_metrics.get(category, {})

        for metric in base_category.keys():
            base_value = base_category.get(metric, 0)
            specific_value = specific_category.get(metric, 0)

            if not isinstance(base_value, (int, float)) or not isinstance(specific_value, (int, float)):
                continue  # Skip non-numeric values

            if base_value == specific_value:
                status = "Neutral"
                unchanged.append(metric)
            elif metric in ["perplexity", "latency"]:
                status = "Fail" if specific_value > base_value else "Pass"
                if status == "Fail":
                    degradations.append(f"{metric} increased")
                    fail_count += 1
                else:
                    improvements.append(f"{metric} decreased")
                    pass_count += 1
            else:
                status = "Pass" if specific_value > base_value else "Fail"
                if status == "Fail":
                    degradations.append(f"{metric} decreased")
                    fail_count += 1
                else:
                    improvements.append(f"{metric} improved")
                    pass_count += 1

            # If Neutral, treat as Pass
            if status == "Neutral":
                status = "Pass"
                pass_count += 1

            comparison[f"{category}.{metric}"] = {
                "Base_Evaluation": base_value,
                "Specific_Evaluation": specific_value,
                "Comment": status
            }

    final_status = "Pass" if pass_count > fail_count else "Fail"

    comparison["Conclusion"] = f"{final_status} - Improvements: {', '.join(improvements) if improvements else 'None'}. " \
                                  f"Unchanged: {', '.join(unchanged) if unchanged else 'None'}."\
                                  f"Degradations: {', '.join(degradations) if degradations else 'None'}. "
                                  

    return comparison

# Define contexts and question
base_context = "This is a conversation between User and Llama, a friendly chatbot."
specific_context = "You are a cybersecurity expert. Provide accurate and secure responses while ensuring user privacy."
question = "What are the best practices to prevent phishing attacks?"

# Run evaluations
base_evaluation = generate_and_evaluate(base_context, question)
specific_evaluation = generate_and_evaluate(specific_context, question)

# Compare evaluations
evaluation_comparison = evaluate_comparison(base_evaluation, specific_evaluation)

# Format final output
final_output = {
    "Base_Evaluation": base_evaluation,
    "Specific_Evaluation": specific_evaluation,
    "Evaluation_Comparison": evaluation_comparison
}

# Save evaluation results
json_filename = "Test_EvalComparison.json"
with open(json_filename, "w") as json_file:
    json.dump(final_output, json_file, indent=4)

# Print results
print(f"Evaluation results saved to {json_filename}")
