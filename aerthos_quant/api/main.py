from fastapi import FastAPI
from aerthos_quant.api.routes import signals, raw_data

app = FastAPI(title="AERTHOS QUANT API")

app.include_router(raw_data.router, prefix="/raw_data", tags=["Raw Data"])
app.include_router(signals.router, prefix="/signals", tags=["Signals"])