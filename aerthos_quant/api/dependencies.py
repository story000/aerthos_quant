from .supabase import get_supabase_client
from .supabase import Client

def provide_supabase_client() -> Client:
    return get_supabase_client()