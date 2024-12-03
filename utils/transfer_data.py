import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

FILE_PATH = r"/media/martin/DATA/_ML/RemoteProject/FinetuneWhisper-Server/temp/11metadata.csv"


@app.get("/download")
async def download_file():
    if os.path.exists(FILE_PATH):
        return FileResponse(FILE_PATH,
                            media_type="application/octet-stream",
                            filename=os.path.basename(FILE_PATH))
    else:
        return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)