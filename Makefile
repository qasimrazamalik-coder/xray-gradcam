PY ?= python

.PHONY: install lint format test smoke data train eval gradcam app clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

# End-to-end run on a tiny synthetic fixture. Proves the pipeline works; numbers are not results.
smoke:
	$(PY) -m xray_gradcam.train --config configs/smoke.yaml
	$(PY) -m xray_gradcam.evaluate --config configs/smoke.yaml
	$(PY) -m xray_gradcam.examples --config configs/smoke.yaml

data:
	$(PY) scripts/download_data.py

train:
	$(PY) -m xray_gradcam.train --config configs/default.yaml
	$(PY) -m xray_gradcam.train --config configs/efficientnet_b0.yaml

eval:
	$(PY) -m xray_gradcam.evaluate --config configs/default.yaml
	$(PY) -m xray_gradcam.evaluate --config configs/efficientnet_b0.yaml

gradcam:
	$(PY) -m xray_gradcam.examples --config configs/default.yaml
	$(PY) -m xray_gradcam.examples --config configs/efficientnet_b0.yaml

app:
	$(PY) app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info results/smoke checkpoints/smoke* data/synthetic_xray
