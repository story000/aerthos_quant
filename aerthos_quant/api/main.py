from fastapi import FastAPI
from aerthos_quant.api.routes import users, signals

app = FastAPI(title="AERTHOS QUANT API")

app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(signals.router, prefix="/signals", tags=["Signals"])