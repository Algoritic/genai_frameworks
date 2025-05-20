import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.index import app

# app = FastAPI()


# @app.post("/callback")
# # echo the payload
# async def callback(data: dict):
#     return JSONResponse(content=data)


client = TestClient(app)


def test_extraction():
    root_path = Path(os.path.dirname(os.path.abspath(__file__)))
    test_payload_path = root_path / "test_payload.pdf"

    with open(test_payload_path, "rb") as f:
        test_schema = {
            "name": "test",
            "description": "test",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        }

        # convert file to base64
        import base64

        base64_file = base64.b64encode(f.read()).decode("utf-8")
        response = client.post(
            "/extract",
            data={
                "files": [base64_file],
                "json_schema": test_schema,
                "callback_url": "http://localhost:8080/callback",
            },
        )
        assert response.status_code == 202
