"""Trusted PositionPredictor checkpoint loading compatibility; current evaluation is cloth_evaluate."""
from ..data_process import DataProcessor
from .model import PositionPredictor
from .train import MODEL_KEYS, read_checkpoint


def load_model(path, device):
    state = read_checkpoint(path)
    config = state["config"]
    model = PositionPredictor(**{k: config[k] for k in MODEL_KEYS}).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, DataProcessor.from_state_dict(state["processor"]), config
