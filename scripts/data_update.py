from aerthos_quant.utils import carbon_loader, obs_loader
from aerthos_quant.predictions import predict
from aerthos_quant.utils.analysis import plot_predictions

if __name__ == "__main__":
    carbon_loader.load_to_supabase()
    obs_loader.load_to_supabase()
    predict.predict_prices_from_file('data/processed/predictions.csv')
    plot_predictions('data/processed/predictions.csv', 'aerthos_quant/predictions/predictions_plot.png')