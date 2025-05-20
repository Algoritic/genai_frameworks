import asyncio
import functools
import http
import os
from pathlib import Path
from typing import Annotated

import json_repair
import jsonschema
import orjson
from fastapi import APIRouter, BackgroundTasks, Form
from promptflow.client import load_flow

from api.utils import PROMPTFLOW_FOLDER, send_webhook
from pf.logger import logger

extraction_router = APIRouter(
    prefix="/extract",
    tags=["extraction"],
    responses={404: {"description": "Not found"}},
)


async def load_and_run_flow(flow_path, webhook_url, file, json_schema, unique_id):
    # Load the flow and run it
    loop = asyncio.get_event_loop()
    f = await loop.run_in_executor(
        None, load_flow, flow_path
    )  # Load flow in a separate thread
    result = await loop.run_in_executor(
        None, functools.partial(f, input_file=file, json_schema=json_schema)
    )
    result["uniqueID"] = unique_id

    if webhook_url:
        logger.debug(f"Sending result to webhook: {webhook_url}")
        await send_webhook(webhook_url, result)


@extraction_router.post(
    "/", tags=["document-extractor"], status_code=http.HTTPStatus.ACCEPTED
)
# files in list of base64 strings
async def extract_document(
    files: Annotated[list[str], Form()],
    callback_url: Annotated[str, Form()] = None,
    background_tasks: BackgroundTasks = None,
    json_schema: Annotated[str, Form()] = None,
    unique_id: Annotated[str, Form()] = None,
    skip_ocr: Annotated[bool, Form()] = False,
):
    if not files or len(files) == 0:
        return {"error": "No files provided"}

    if json_schema is None:
        return {"error": "No json schema provided"}

    json_data = json_repair.loads(json_schema)
    json_string = orjson.dumps(json_data)

    try:
        jsonschema.Draft202012Validator.check_schema(json_repair.loads(json_string))
    except jsonschema.SchemaError as e:
        return {"error": f"Invalid JSON schema: {str(e)}"}

    root_path = (
        Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        / PROMPTFLOW_FOLDER
    )

    flow_path = None
    if skip_ocr:
        flow_path = os.path.join(root_path, "extraction.dag.yaml")
    else:
        flow_path = os.path.join(root_path, "flow.dag.yaml")

    for file in files:
        background_tasks.add_task(
            load_and_run_flow, flow_path, callback_url, file, json_data, unique_id
        )

    return {"message": "Accepted", "status": 202}
