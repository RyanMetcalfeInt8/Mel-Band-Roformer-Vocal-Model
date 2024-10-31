import time
import numpy as np
import torch
import sys
import torch.nn as nn


def get_model_from_config(model_type, config):
    if model_type == 'mel_band_roformer':
        from models.mel_band_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)

def demix_track(config, model, mix, device, first_chunk_time=None):
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    step = C // N
    fade_size = C // 10
    border = C - step

    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    windowing_array = get_windowing_array(C, fade_size, device)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

            i = 0
            total_length = mix.shape[1]
            num_chunks = (total_length + step - 1) // step

            if first_chunk_time is None:
                start_time = time.time()
                first_chunk = True
            else:
                start_time = None
                first_chunk = False

            while i < total_length:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

                if first_chunk and i == 0:
                    chunk_start_time = time.time()

                model_in = part.unsqueeze(0)
                print("model_in.shape = ", model_in.shape)

                print("type of model = ", type(model))
                if i == 0 and False:
                    from openvino.tools import mo
                    from openvino.runtime import serialize
                    print("converting to onnx...")
                    torch.onnx.export(model, (model_in), "model.onnx", input_names=["input"], output_names=["output"])
                    print("conversion to onnx done..")

                if False:
                    model_out = model(model_in)
                else:
                    stft_repr, stft_window, istft_length = model.pre_forward(model_in)

                    if i == 0 and False:
                        print("converting to onnx...")
                        torch.onnx.export(model, (stft_repr), "mel_band_reformer.onnx", input_names=["stft_repr"], output_names=["recon_audio", "masks"])
                        print("conversion to onnx done..")

                    stft_repr, masks = model(stft_repr)
                    model_out = model.post_forward(stft_repr, masks, stft_window, istft_length)


                print("model_out.shape = ", model_out.shape)


                print("converting to csm to onnx model..")
                x = model_out[0]

                window = windowing_array.clone()
                if i == 0:
                    window[:fade_size] = 1
                elif i + C >= total_length:
                    window[-fade_size:] = 1

                result[..., i:i+length] += x[..., :length] * window[..., :length]
                counter[..., i:i+length] += window[..., :length]
                i += step

                if first_chunk and i == step:
                    chunk_time = time.time() - chunk_start_time
                    first_chunk_time = chunk_time
                    estimated_total_time = chunk_time * num_chunks
                    print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
                    first_chunk = False

                if first_chunk_time is not None and i > step:
                    chunks_processed = i // step
                    time_remaining = first_chunk_time * (num_chunks - chunks_processed)
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds")
                    sys.stdout.flush()

            print()
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if mix.shape[1] > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}, first_chunk_time
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}, first_chunk_time
