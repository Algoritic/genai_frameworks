import asyncio
import functools
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Form
from promptflow.client import load_flow

from api.utils import PROMPTFLOW_FOLDER, send_webhook

classification_router = APIRouter(
    prefix="/classify",
    tags=["classification"],
    responses={404: {"description": "Not found"}},
)


async def load_and_run_flow(flow_path, webhook_url, file, unique_id):
    # Load the flow and run it

    try:
        loop = asyncio.get_event_loop()
        f = await loop.run_in_executor(
            None, load_flow, flow_path
        )  # Load flow in a separate thread
        result = await loop.run_in_executor(
            None,
            functools.partial(f, input_file=file, callback_url=webhook_url),
        )
        result["uniqueID"] = unique_id

        if webhook_url:
            await send_webhook(webhook_url, result)
    except Exception as e:
        print(f"Error loading and running flow: {e}")
        if webhook_url:
            await send_webhook(webhook_url, {"error": str(e)})


@classification_router.post("/", tags=["document-classification"], status_code=200)
async def classify_document(
    files: Annotated[list[str], Form()],
    background_tasks: BackgroundTasks = None,
    callback_url: Annotated[str, Form()] = None,
):
    """
    Classify documents using a classification model.
    """
    if not files or len(files) == 0:
        return {"error": "No files provided"}

    root_path = (
        Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        / PROMPTFLOW_FOLDER
    )

    flow_path = os.path.join(root_path, "classification.dag.yaml")

    for file in files:
        background_tasks.add_task(
            load_and_run_flow, flow_path, callback_url, file, None
        )
    return {"message": "Accepted", "status": 202}
