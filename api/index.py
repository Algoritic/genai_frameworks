from fastapi import FastAPI
from routes.extraction import extraction_router
from routes.classification import classification_router
from routes.comparison import comparison_router
from promptflow.tracing import start_trace


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

    return app


start_trace()
app = create_app()
