"""Offline CPU smoke tests for the spectral-evidence framework.

Install requirements.txt, then run from the repository root:
    python -B -m unittest discover -s smoke_tests -v

Tests use synthetic inputs and a randomly initialized tiny language model.
No model weights or datasets are downloaded. Full pretrained-model training,
CUDA/DDP execution and external benchmark integrations are outside this suite.
"""
