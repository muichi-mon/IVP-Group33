PY ?= python
CONFIGS = configs/convnext_tiny.yaml configs/efficientnetv2_s.yaml configs/resnet50d.yaml

.PHONY: help install clean-aug folds train-convnext train-effnet train-resnet train-vit \
        train-all predict-convnext predict-effnet predict-resnet predict-vit predict-all \
        ensemble pseudo pseudo-round submission smoke

help:
	@echo "Targets:"
	@echo "  install         Install Python deps from requirements.txt"
	@echo "  clean-aug       Remove offline-augmented PNGs (*_aug*.png)"
	@echo "  folds           Generate stratified 5-fold split (data/folds.csv)"
	@echo "  train-convnext  Train convnext_tiny across 5 folds"
	@echo "  train-effnet    Train tf_efficientnetv2_s across 5 folds"
	@echo "  train-resnet    Train resnet50d across 5 folds"
	@echo "  train-all       Run all three trainers sequentially"
	@echo "  predict-all     TTA inference for every fold/arch checkpoint"
	@echo "  ensemble        Average all preds/*.npy -> submission.csv"
	@echo "  pseudo          Generate data/pseudo.csv from current preds"
	@echo "  pseudo-round    Re-train convnext with pseudo labels and re-ensemble"
	@echo "  submission      Alias for predict-all + ensemble"
	@echo "  smoke           2-epoch sanity-check run on convnext fold 0"

install:
	$(PY) -m pip install -r requirements.txt

clean-aug:
	$(PY) augment_data.py --clean

folds:
	$(PY) -m src.folds

train-convnext: folds
	$(PY) -m src.run_kfold --config configs/convnext_tiny.yaml

train-effnet: folds
	$(PY) -m src.run_kfold --config configs/efficientnetv2_s.yaml

train-resnet: folds
	$(PY) -m src.run_kfold --config configs/resnet50d.yaml

train-vit: folds
	$(PY) -m src.run_kfold --config configs/vit_small.yaml

train-all: train-convnext train-effnet train-resnet train-vit

predict-convnext:
	@for f in 0 1 2 3 4; do $(PY) -m src.predict --config configs/convnext_tiny.yaml --fold $$f; done

predict-effnet:
	@for f in 0 1 2 3 4; do $(PY) -m src.predict --config configs/efficientnetv2_s.yaml --fold $$f; done

predict-resnet:
	@for f in 0 1 2 3 4; do $(PY) -m src.predict --config configs/resnet50d.yaml --fold $$f; done

predict-vit:
	@for f in 0 1 2 3 4; do $(PY) -m src.predict --config configs/vit_small.yaml --fold $$f; done

predict-all: predict-convnext predict-effnet predict-resnet predict-vit

ensemble:
	$(PY) -m src.ensemble

pseudo:
	$(PY) -m src.pseudo_label --threshold 0.99

pseudo-round: pseudo
	$(PY) -m src.run_kfold --config configs/convnext_tiny.yaml --pseudo data/pseudo.csv
	@for f in 0 1 2 3 4; do $(PY) -m src.predict --config configs/convnext_tiny.yaml --fold $$f; done
	$(PY) -m src.ensemble --out submission_pseudo.csv

submission: predict-all ensemble

smoke:
	$(PY) -m src.train --config configs/convnext_tiny.yaml --fold 0 --debug
