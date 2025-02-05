import os
from pathlib import Path
import tempfile
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from promptflow.client import load_flow

document_classifier_router = APIRouter(
    prefix="/document-classifier",
    tags=["document-classifier"],
    responses={404: {
        "description": "Not found"
    }})


#tags separated by comma
@document_classifier_router.post("/", tags=["document-classifier"])
async def document_classifier(files: Annotated[list[UploadFile],
                                               File()],
                              tags: Annotated[str, Form()]):
    tags_str = tags
    tags = tags.split(",")
    for file in files:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(file.file.read())
            temp_file.seek(0)
            root_path = Path(os.path.dirname(
                os.path.abspath(__file__))).parent.parent
            flow_path = os.path.join(root_path, "classification_flow.dag.yaml")
            f = load_flow(flow_path)
            result = f(input_file=temp_file.name, available_tags=tags_str)
            return {
                "file_name": file.filename,
                "result": result["tagging_result"]
            }
