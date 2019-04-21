import sys
import json
sys.path.append('./src')

from libs.EvaluateModels import EvaluateModels

params = {
    "approach": "Engine_summary_reggresion",
    "regenarate": True,
    "train_cutoff": True,
    "num_resample_train": 1,
    "feature": [
        "last_dur",
        "timegrad",
        "all_col_max"
    ],
    "scaling": True,
    "use_model": "SVR",
    "model_params": {
        "gamma": "auto"
    }
}

test = EvaluateModels(params)
test.run_cv()

params = {
    "approach": "Engine_summary_reggresion",
    "regenarate": True,
    "train_cutoff": True,
    "num_resample_train": 1,
    "feature": [
        "last_dur",
        "timegrad",
        "all_col_min"
    ],
    "scaling": True,
    "use_model": "SVR",
    "model_params": {
        "gamma": "auto"
    }
}

test = EvaluateModels(params)
test.run_cv()

params = {
    "approach": "Engine_summary_reggresion",
    "regenarate": True,
    "train_cutoff": True,
    "num_resample_train": 1,
    "feature": [
        "last_dur",
        "timegrad",
        "all_col_mean"
    ],
    "scaling": True,
    "use_model": "SVR",
    "model_params": {
        "gamma": "auto"
    }
}

test = EvaluateModels(params)
test.run_cv()

params = {
    "approach": "Engine_summary_reggresion",
    "regenarate": True,
    "train_cutoff": True,
    "num_resample_train": 1,
    "feature": [
        "last_dur",
        "timegrad",
        "all_col_std"
    ],
    "scaling": True,
    "use_model": "SVR",
    "model_params": {
        "gamma": "auto"
    }
}

test = EvaluateModels(params)
test.run_cv()

params = {
    "approach": "Engine_summary_reggresion",
    "regenarate": True,
    "train_cutoff": True,
    "num_resample_train": 1,
    "feature": [
        "last_dur",
        "timegrad",
        "timegrad_sted"
    ],
    "scaling": True,
    "use_model": "SVR",
    "model_params": {
        "gamma": "auto"
    }
}

test = EvaluateModels(params)
test.run_cv()

