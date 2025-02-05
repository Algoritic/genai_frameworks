# Start API

uvicorn src.api.index:app --host 0.0.0.0 --port 8080 --reload

# API Endpoints

POST /document-parser
files: File Attachment, output_format: json
POST /document-classification
files: File Attachment, tags: available tags

# Promptflow Engine

pf run create -f ./src/run.yml
pf flow test --flow ./src/flow.dag.yaml
pf run create --flow ./src/flow.flex.yaml --data ./src/data_eval.jsonl --stream
