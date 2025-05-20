`pf flow test --flow tools.extract_json:extract_json --inputs ocr_output="Paris" schema="{\"name\":\"capital\",\"strict\":false,\"schema\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\",\"description\":\"Name of the country\"}}}}"`

`uvicorn api.index:app --host 0.0.0.0 --port 8080 --reload --limit-max-requests 50000000`

`pytest ./test/e2e.py`
