from aerthos_quant.utils import carbon_loader, obs_loader

if __name__ == "__main__":
    carbon_loader.load_to_supabase()
    obs_loader.load_to_supabase()