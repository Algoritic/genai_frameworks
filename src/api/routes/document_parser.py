import asyncio
import functools
import http
import json
import os
from pathlib import Path
import tempfile
from typing import Annotated
from fastapi import APIRouter, File, Form, UploadFile, BackgroundTasks
import httpx
from promptflow.client import load_flow
import json_repair

# from src.processors.file_processor import batch_optimize_image, pdf_to_images
# from src.tools.ocr import simple_pdf_ocr, use_easy_ocr
import jsonschema

document_parser_router = APIRouter(
    prefix="/document-parser",
    tags=["document-parser"],
    responses={404: {
        "description": "Not found"
    }})


async def send_webhook(webhook_url, result):
    async with httpx.AsyncClient() as client:
        print("sending webhook")
        await client.post(webhook_url, json={"result": result})


async def load_and_run_flow(flow_path,
                            webhook_url,
                            file_name,
                            output_format="json",
                            json_schema=None):
    loop = asyncio.get_running_loop()
    f = await loop.run_in_executor(None, load_flow,
                                   flow_path)  # Load flow in a separate thread
    result = await loop.run_in_executor(
        None,
        functools.partial(f,
                          input_file=file_name,
                          use_schema=False if json_schema is None else True,
                          json_schema=json_schema)
    )  # Execute the loaded function concurrently

    #delete the file
    os.remove(file_name)

    # Call the webhook with the result
    await send_webhook(webhook_url, result)


async def process_file(file: UploadFile,
                       flow_path: str,
                       background_tasks: BackgroundTasks,
                       output_format: str,
                       json_schema: str = None,
                       callback_url: str = None):
    file_ext = file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=f".{file_ext}") as temp_file:
        temp_file.write(file.file.read())
        temp_file.seek(0)

        background_tasks.add_task(
            load_and_run_flow, flow_path,
            "https://webhook.site/65185ced-e3a3-4c16-bc1f-c4769e17aa47",
            temp_file.name, output_format, json_schema)


#output format: json, key-value
@document_parser_router.post("/",
                             tags=["document-parser"],
                             status_code=http.HTTPStatus.ACCEPTED)
async def document_parser(
        files: Annotated[list[UploadFile], File()],
        background_tasks: BackgroundTasks,
        output_format: Annotated[str, Form()] = "json",
        #optional json_schema
        callback_url: Annotated[str, Form()] = None,
        json_schema: Annotated[str, Form()] = None,
        unique_identifier: Annotated[str, Form()] = None):
    if not files or len(files) == 0:
        return {"message": "No upload files sent"}

    if json_schema is not None:
        json_data = json_repair.loads(json_schema)
        json_string = json.dumps(json_data, separators=(",", ":"))

        try:
            jsonschema.Draft202012Validator.check_schema(
                json_repair.loads(json_string))
        except jsonschema.SchemaError as e:
            return {"message": "Invalid JSON schema: " + str(e)}
    else:
        json_string = None

    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    flow_path = os.path.join(root_path, "flow.dag.yaml")
    tasks = [
        asyncio.create_task(
            process_file(file, flow_path, background_tasks, output_format,
                         json_string, callback_url)) for file in files
    ]
    await asyncio.gather(*tasks)

    # for file in files:
    #     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #         temp_file.write(file.file.read())
    #         temp_file.seek(0)

    #         root_path = Path(os.path.dirname(
    #             os.path.abspath(__file__))).parent.parent
    #         flow_path = os.path.join(root_path, "flow.dag.yaml")
    #         background_tasks.add_task(
    #             load_and_run_flow, flow_path,
    #             "https://webhook.site/04e1fcfa-0433-4db2-9817-a4d761480aab",
    #             temp_file.name, output_format)

    return {"message": "Accepted", "status": 202}

    # f = load_flow(flow_path)
    # result = f()

    # for file in files:
    #     folder_path = pdf_to_images(file.file.read())
    #     folder_path = batch_optimize_image(folder_path)
    #     page_text = ""
    #     for f in os.listdir(folder_path):
    #         with open(f"{folder_path}/{f}", "rb") as f:
    #             page_text += use_easy_ocr(f.read())

    #     print(page_text)

    return {"message": "success"}
