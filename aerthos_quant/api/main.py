from fastapi import FastAPI
from aerthos_quant.api.routes import signals, data

app = FastAPI(title="AERTHOS QUANT API")

app.include_router(data.router, prefix="/data/{param}", tags=["Raw Data"])
app.include_router(signals.router, prefix="/signals", tags=["Signals"])