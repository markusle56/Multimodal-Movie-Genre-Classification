#!/usr/bin/env bash
# Reproduce every result in outputs/results/.
# preprocess -> featurise -> preprocessing ablation -> tuning (validation only) -> train -> decision study
# -> analysis -> duplicate audit -> EDA notebook (executed in place).
# Usage: bash run_pipeline.sh            (uses `python` on PATH; set PY=/path/to/python to override)
# Runtime on an Apple M3 (16 GB): about 3 hours, most of it feature extraction and cross-attention training.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"

test -f data/data.npy -a -f data/images.npz || {
  echo "Dataset missing. Download it with:"
  echo "  kaggle datasets download -d javierurea/simplified-mm-imdb -p data --unzip"
  exit 1; }

"$PY" scripts/01_preprocess.py                                   # split, duplicate groups, 5 text variants

# features: text variants + the two poster encodings compared in the ablation
"$PY" scripts/02_featurise.py --only distil clip_img --preps crop lbox
"$PY" scripts/03_ablation.py                                     # choose text cleaning on validation
V=$("$PY" -c "import json;print(json.load(open('outputs/splits/choices.json'))['distil_variant'])")

# features for the tuning study and the final models
"$PY" scripts/02_featurise.py --only distil --variants "$V" --tokens                 # plot chunk tokens
"$PY" scripts/02_featurise.py --only distil --variants "$V" --distil-len 512         # context study
"$PY" scripts/02_featurise.py --only clip_img --preps crop multi --tokens            # patch tokens (grid / final)
"$PY" scripts/02_featurise.py --only clip_img --preps native                         # native-aspect encoding
"$PY" scripts/02_featurise.py --only resnet vit                                      # ImageNet encoders
"$PY" scripts/02_featurise.py --only bert clip_txt --variants "$V"                   # text encoder comparison

"$PY" scripts/04_tuning.py --part search                         # A-D, G on validation; updates choices.json
"$PY" scripts/05_train.py                                        # 13 configurations x 3 seeds
"$PY" scripts/04_tuning.py --part decision                       # thresholds and late-fusion weights (seed 0)
"$PY" scripts/06_analyse.py                                      # significance, robustness, bias -> analysis.json
"$PY" scripts/07_duplicates.py                                   # near-duplicate audit + leakage simulation

"$PY" -m jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb --ExecutePreprocessor.timeout=1800
echo "Done -> outputs/results/"
