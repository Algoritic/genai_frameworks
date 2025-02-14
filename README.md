# AI Model Evaluation

This project provides a web-based interface using Streamlit to evaluate AI-generated responses by comparing a LLM model with base prompt and a LLM model with a custom prompt. The evaluation includes various quantitative and qualitative metrics such as perplexity, coherence, readability, and safety.

## Installation

Ensure you have Python installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Navigate to the `app` directory:

```sh
cd app
```

2. Run the Streamlit application:

```sh
streamlit run app.py
```

3. Open the provided local URL in your browser.

## Features

- **Perplexity Calculation**: Measures how well the AI model predicts the next word.
- **Confidence Score**: Estimates response confidence based on length.
- **Relevance Score**: Computes semantic similarity between the question and response.
- **Coherence & Readability**: Evaluates logical flow and ease of understanding.
- **Bias & Safety Checks**: Identifies potential biases and unsafe content.
- **Latency Measurement**: Calculates response time.
- **Comparison Analysis**: Assesses improvements or degradations in model performance.

## Model Evaluation Workflow

1. Input the **Model Server URL**.
2. Provide a **Base Prompt** and a **Specific Prompt**.
3. Enter a **Test Question**.
4. Click **Run Evaluation** to compare the AI models.

## Dependencies

The application uses the following libraries:

- `streamlit` - Web framework for UI.
- `requests` - Handles API calls.
- `torch` - Required for PyTorch models.
- `transformers` - Loads GPT-2 and other NLP models.
- `sentence-transformers` - Computes semantic similarity.
- `bert-score` - Evaluates coherence between sentences.
- `textstat` - Calculates readability scores.
- `detoxify` - Detects potential toxicity in responses.

## Notes

- Ensure the AI model is running on the specified **Model Server URL**.
- Adjust the prompts for different domain-specific evaluations.
- The evaluation script supports models like **LLaMA_CPP** and similar chat-based AI systems.

## Contact

For any issues or improvements, feel free to open an issue or contribute to the repository.

