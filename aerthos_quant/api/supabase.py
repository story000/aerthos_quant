from supabase import create_client, Client
from dotenv import load_dotenv
import os
load_dotenv()
_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not _SUPABASE_URL or not _SUPABASE_KEY:
    raise ValueError("Supabase URL 和 Key 未正确配置，请检查环境变量。")


_client: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

def get_supabase_client() -> Client:
    return _client

def fetch_table(table_name: str) -> list[dict]:
    """
    获取指定表的数据
    
    参数:
        table_name: 表名
    
    返回:
        表数据列表
    """
    query = _client.table(table_name).select("*")
    
    return query.execute().data

def insert_data(table_name: str, data: dict):
    return _client.table(table_name).insert(data).execute().data
