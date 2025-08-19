# Neural Network Analysis on Silero VAD

This directory contains some analysis on Silero VAD. The motivation is to
understand how a VAD works.

## List of Contents

`silero.py`

* PyTorch implementation of SileroVAD
* Re-implemented by hand based on the original TorshScript version.

`torchscript_to_safetensors.py`

* Extract tensor data from TorchScript file.

`silero-vad/data/silero_vad.safetensors`

* Pre-trained weight & bias tensors.

## Network Architecture

![Network Analysis of SileroVAD](blueprint.png)
