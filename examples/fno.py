"""Dataset and FNO constructors used in the workshop.

Adapted from yang926/AI4Sci-PhysicsNeMo-Bootcamp,
fno_physicsnemo_l1.py.
"""
from torch.utils.data import TensorDataset
from physicsnemo.models.fno import FNO

def build_datasets(train_pairs, val_pairs, test_pairs):
    return (TensorDataset(*train_pairs), TensorDataset(*val_pairs),
            TensorDataset(*test_pairs))

def build_model(model_config, grid_size):
    return FNO(in_channels=1, out_channels=1, dimension=2, **model_config)
