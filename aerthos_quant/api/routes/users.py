from fastapi import APIRouter, Depends
from supabase import Client
from aerthos_quant.api.dependencies import provide_supabase_client

router = APIRouter()

@router.get("/")
def list_users(supabase: Client = Depends(provide_supabase_client)):
    """
    返回 users 表中的所有用户信息。
    该函数使用 Supabase 客户端从数据库中获取用户数据。
    """
    result = supabase.table("User").select("*").execute()
    return result.data
