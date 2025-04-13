train:
	python scripts/train_model.py

backtest:
	python scripts/run_backtest.py

api:
	uvicorn aerthos_quant.api.main:app --reload