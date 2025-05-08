import os
import requests
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/callback")
#echo the payload
async def callback(data: dict):
    return JSONResponse(content=data)
