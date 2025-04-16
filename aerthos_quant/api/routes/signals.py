from fastapi import APIRouter, Query
from typing import List, Optional
from aerthos_quant.strategies.signal_generator import Signal_Generator
import os
from fastapi.responses import JSONResponse
router = APIRouter()

@router.get("/")
def get_signals():
    df = Buy_Signal_Generator()
    df['Date'] = df.index.strftime("%Y-%m-%d")
   
    return df.fillna("").to_dict(orient="records")


@router.get("/list_pics")
def list_pictures():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "pics"))
    if not os.path.exists(base_dir):
        return JSONResponse([], status_code=200)
    filenames = [f for f in os.listdir(base_dir) if f.endswith(".png")]
    return JSONResponse(filenames)
