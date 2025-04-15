from fastapi import FastAPI
from aerthos_quant.api.routes import signals, data, predictions
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
app = FastAPI(title="AERTHOS QUANT API")
predictions_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "predictions"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return {"message": "pong"}

app.include_router(data.router, prefix="/data/{param}", tags=["Raw Data"])
app.include_router(signals.router, prefix="/signals", tags=["Signals"])
app.mount("/static", StaticFiles(directory=predictions_dir), name="static")
