import httpx


async def send_webhook(webhook_url, result):
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json={"result": result})


PROMPTFLOW_FOLDER = "pf"
