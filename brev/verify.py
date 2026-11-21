"""Verify selected workshop imports and real CUDA compute, without training."""
from importlib.metadata import version
import sys
import torch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.eq.pde import PDE
import h5py
import matplotlib
import numpy
import sympy
import yaml
import ipykernel

assert sys.version_info[:3] == (3, 12, 11), "Expected Python 3.12.11"
assert version("nvidia-physicsnemo") == "2.2.2"
assert torch.__version__ == "2.10.0+cu128"
assert version("torchvision") == "0.25.0+cu128"
assert torch.cuda.is_available(), "CUDA unavailable; inspect the VM GPU and driver"
value = torch.tensor([1.0], device="cuda", requires_grad=True)
value.square().sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(value.grad).all(), "CUDA derivative check failed"
print("Workshop imports and CUDA compute verified:", torch.cuda.get_device_name(0))
