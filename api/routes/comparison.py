import os
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Form
import json_repair
from promptflow.client import load_flow
from promptflow.tracing import trace

from utils import PROMPTFLOW_FOLDER

comparison_router = APIRouter(
    prefix="/compare",
    tags=["comparison"],
    responses={404: {
        "description": "Not found"
    }},
)


@trace
@comparison_router.post("/", tags=["document-comparison"], status_code=200)
async def compare(key_sets: Annotated[str, Form()]):

    json_data = json_repair.loads(key_sets)
    root_path = Path(os.path.dirname(
        os.path.abspath(__file__))).parent.parent / PROMPTFLOW_FOLDER
    flow_path = os.path.join(root_path, "comparison.dag.yaml")
    flow = load_flow(flow_path)
    comparison_result = flow(comparison_key_sets=json_data)
    return comparison_result
