train:
	python scripts/train_model.py

backtest:
	python scripts/run_backtest.py

api:
	uvicorn aerthos_quant.api.main:app --reload --host 0.0.0.0 --port 8000

update:
	PYTHONPATH=. python scripts/data_update.py
