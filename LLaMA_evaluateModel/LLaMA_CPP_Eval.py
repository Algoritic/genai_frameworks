import requests
import json
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from bert_score import score  # For coherence scoring

# Load models
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define Model Server URL
MODEL_SERVER_URL = "http://localhost:8080/v1/chat/completions" # please download the model at 'https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/blob/main/Llama-3.2-1B-Instruct.Q6_K.llamafile?download=true'

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
    return round(min(len(response) / 50, 1.0), 2)  # Normalize between 0 and 1

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
        return 1.00  # Single sentence, assume coherent
    
    _, _, coherence_score = score(sentences[:-1], sentences[1:], lang="en", rescale_with_baseline=True)
    return round(coherence_score.mean().item(), 2)

# def compute_hallucination_rate(response, knowledge_base):
#     """Compute hallucination rate by comparing response with trusted knowledge(REQUIRE KNOWLEDGE BASE)"""
#     """Higher score means higher hallucination risk"""
#     response_embedding = embedding_model.encode(response, convert_to_tensor=True)
#     knowledge_embeddings = [embedding_model.encode(kb, convert_to_tensor=True) for kb in knowledge_base]
    
#     max_similarity = max(util.pytorch_cos_sim(response_embedding, kb_emb).item() for kb_emb in knowledge_embeddings)
    
#     return 1 - max_similarity  # Higher score means higher hallucination risk

# def generate_and_evaluate(context, question, knowledge_base):
def generate_and_evaluate(context, question):
    """Generates response from LLM, evaluates it, and returns evaluation metrics."""
    request_payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ]
    }

    start_time = time.time()
    response = requests.post(MODEL_SERVER_URL, json=request_payload, timeout=60)
    latency = round((time.time() - start_time), 2)  # Measure response time

    if response.status_code == 200:
        response_data = response.json()
        generated_text = response_data["choices"][0]["message"]["content"]

        # Extract token usage statistics
        token_usage = response_data.get("usage", {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        })

        # Compute evaluation metrics
        perplexity = compute_perplexity(generated_text)
        confidence = compute_heuristic_confidence(generated_text)
        relevance = compute_relevance(question, generated_text)
        coherence = compute_coherence(generated_text)
        # hallucination_rate = compute_hallucination_rate(generated_text, knowledge_base)

        return {
            "user_question":question,
            "context": context,
            "llm_response": generated_text,
            "usage": token_usage,
            "evaluation_metrics": {
                "perplexity": perplexity,
                "confidence": confidence,
                "relevance": relevance,
                "coherence": coherence,
                # "hallucination_rate": hallucination_rate,
                "latency": latency
            }
        }
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def evaluate_comparison(base_eval, specific_eval):
    """Compares evaluation metrics and determines Pass/Fail status with reason."""
    comparison = {}
    final_status = "Pass"
    final_reason = []
    
    base_metrics = base_eval.get("evaluation_metrics", {})
    specific_metrics = specific_eval.get("evaluation_metrics", {})

    for metric, base_value in base_metrics.items():
        specific_value = specific_metrics.get(metric)

        if isinstance(base_value, dict) and isinstance(specific_value, dict):
            # Recursively evaluate nested metrics
            comparison[metric] = evaluate_comparison({"evaluation_metrics": base_value}, 
                                                     {"evaluation_metrics": specific_value})
            continue
        
        if specific_value is None:
            continue  # Skip missing metrics

        if base_value == specific_value:
            status = "No Change"
        elif metric in ["perplexity", "latency"]:
            status = "Fail" if specific_value > base_value else "Pass"
            if status == "Fail":
                final_reason.append(f"{metric} increased")
        else:
            status = "Pass" if specific_value > base_value else "Fail"
            if status == "Fail":
                final_reason.append(f"{metric} decreased")

        comparison[metric] = {
            "Base_Evaluation": base_value,
            "Specific_Evaluation": specific_value,
            "Comment": status
        }
    
    if any(comparison[metric]["Comment"] == "Fail" for metric in comparison if isinstance(comparison[metric], dict)):
        final_status = "Fail"

    comparison["Final_Comment"] = f"{final_status} - {', '.join(final_reason) if final_reason else 'No significant degradation.'}"

    return comparison

# Debugging: Print evaluation structures before comparison
# print(json.dumps(base_evaluation, indent=4))
# print(json.dumps(specific_evaluation, indent=4))

# Define contexts and question
base_context = "This is a conversation between User and Llama, a friendly chatbot."
specific_context = "You are a technical solution analyst. Please answer user question correctly and avoid giving false answers."
question = "How do we fix hallucination in AI?"

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
json_filename = "LLaMACPP_EvalComparison.json"
with open(json_filename, "w") as json_file:
    json.dump(final_output, json_file, indent=4)

# Print results
print(f"Evaluation results saved to {json_filename}")