from fastapi import FastAPI
from promptflow.tracing import start_trace

from api.routes.classification import classification_router
from api.routes.comparison import comparison_router
from api.routes.extraction import extraction_router
from pf.logger import logger


def create_app():
    app = FastAPI(
        title="Document Extraction API",
        description="API for document extraction using PromptFlow",
        version="1.0.0",
        openapi_tags=[
            {
                "name": "document-extractor",
                "description": "Document extraction using PromptFlow",
            },
        ],
    )

    # Include the router
    app.include_router(extraction_router)
    app.include_router(classification_router)
    app.include_router(comparison_router)

    # add echo route
    @app.post("/callback")
    # echo the payload
    async def callback(data: dict):
        logger.debug(f"Received callback data: {data}")
        return data

    return app


start_trace()
app = create_app()
