import requests
import json
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

# Load models for evaluation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define Model Server URL
MODEL_SERVER_URL = "http://localhost:8080/v1/chat/completions"

def compute_perplexity(response):
    """Compute perplexity using GPT-2"""
    inputs = tokenizer(response, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

def compute_heuristic_confidence(response):
    """Simple heuristic: longer responses indicate higher confidence."""
    return min(len(response) / 50, 1.0)  # Normalize between 0 and 1

def compute_relevance(question, response):
    """Compute relevance using cosine similarity between embeddings"""
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(question_embedding, response_embedding).item()
    return similarity_score

def generate_and_evaluate(context, question, output_file="LLaMACPP_BaseEval.json"):
    """Generates response from LLM, evaluates it, and saves both in a JSON file."""
    request_payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ]
    }

    start_time = time.time()
    response = requests.post(MODEL_SERVER_URL, json=request_payload, timeout=60)
    latency = time.time() - start_time  # Measure response time

    if response.status_code == 200:
        response_data = response.json()
        generated_text = response_data["choices"][0]["message"]["content"]

        # Compute evaluation metrics
        perplexity = compute_perplexity(generated_text)
        confidence = compute_heuristic_confidence(generated_text)
        relevance = compute_relevance(question, generated_text)
        coherence = relevance  # Approximate coherence using relevance
        hallucination_rate = 0  # Placeholder (requires fact-checking)

        # Combine original LLM response and evaluation
        final_output = {
            "llm_response": response_data,  # Store the original LLM output
            "evaluation_metrics": {
                "perplexity": perplexity,
                "confidence": confidence,
                "relevance": relevance,
                "coherence": coherence,
                "hallucination_rate": hallucination_rate,
                "latency": latency
            }
        }

        # Save the final output to a JSON file
        with open(output_file, "w") as file:
            json.dump(final_output, file, indent=4)

        print(f"Final output saved to {output_file}")
        return final_output
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Test with sample input
context = "This is a conversation between User and Llama, a friendly chatbot."
question = "Hello World!"
generate_and_evaluate(context, question)