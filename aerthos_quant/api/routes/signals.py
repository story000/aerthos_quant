from fastapi import APIRouter, Query
from typing import List, Optional
from aerthos_quant.strategies.signal_generator import generate_signals

router = APIRouter()

@router.get("/")
def get_signals(symbols: Optional[List[str]] = Query(default=None, description="可选的股票代码列表")):
    """
    返回一组交易信号。如果传入 symbols，则只返回相关标的的信号。
    """
    signals = generate_signals(symbols)
    return {"signals": signals}
