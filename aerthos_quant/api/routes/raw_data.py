from fastapi import APIRouter, Depends
from supabase import Client
from aerthos_quant.api.dependencies import provide_supabase_client
from aerthos_quant.api.supabase import fetch_table

router = APIRouter()

@router.get("/carbon")
def carbon(supabase: Client = Depends(provide_supabase_client)):
    """
    返回 carbon 表中的所有碳价格历史数据。
    该函数使用 Supabase 客户端从数据库中获取碳价格历史数据。
    """
    result = fetch_table("carbon")
    return result
