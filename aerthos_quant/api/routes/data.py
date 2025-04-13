from fastapi import APIRouter, Depends
from supabase import Client
from aerthos_quant.api.dependencies import provide_supabase_client
from aerthos_quant.api.supabase import fetch_table

router = APIRouter()

@router.get("/")
def data(param: str, supabase: Client = Depends(provide_supabase_client)):
    """
    返回数据库表格中的所有数据。
    该函数使用 Supabase 客户端从数据库中获取数据。
    """
    result = fetch_table(param.lower())
    return result
