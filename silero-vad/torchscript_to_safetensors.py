"""\
Usage: python torchscript_to_safetensors.py silero_vad.jit

This script extracts weights & biases from silero_vad.jit,
and save them in HuggingFace Safetensors format.
"""

import sys
import torch
import safetensors.torch

def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1

    model_path = sys.argv[1]

    model = torch.jit.load(model_path)

    safetensors.torch.save_file({
        "stft.conv.weight": model._model.stft.forward_basis_buffer,
        "encoder.se.conv0.weight": getattr(model._model.encoder, '0').reparam_conv.weight,
        "encoder.se.conv0.bias": getattr(model._model.encoder, '0').reparam_conv.bias,
        "encoder.se.conv1.weight": getattr(model._model.encoder, '1').reparam_conv.weight,
        "encoder.se.conv1.bias": getattr(model._model.encoder, '1').reparam_conv.bias,
        "encoder.se.conv2.weight": getattr(model._model.encoder, '2').reparam_conv.weight,
        "encoder.se.conv2.bias": getattr(model._model.encoder, '2').reparam_conv.bias,
        "encoder.se.conv3.weight": getattr(model._model.encoder, '3').reparam_conv.weight,
        "encoder.se.conv3.bias": getattr(model._model.encoder, '3').reparam_conv.bias,
        "decoder.lstm.weight_ih_l0": model._model.decoder.rnn.weight_ih,
        "decoder.lstm.bias_ih_l0": model._model.decoder.rnn.bias_ih,
        "decoder.lstm.weight_hh_l0": model._model.decoder.rnn.weight_hh,
        "decoder.lstm.bias_hh_l0": model._model.decoder.rnn.bias_hh,
        "decoder.se.conv.weight": getattr(model._model.decoder.decoder, '2').weight,
        "decoder.se.conv.bias": getattr(model._model.decoder.decoder, '2').bias
    }, "silero_vad.safetensors")

if __name__ == '__main__':
    sys.exit(main())
