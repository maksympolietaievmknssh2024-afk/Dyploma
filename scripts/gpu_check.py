import torch

print('CUDA available:', torch.cuda.is_available())
print('Torch version:', torch.__version__)
print('CUDA version:', getattr(torch.version, 'cuda', None))
print()

if torch.cuda.is_available():
    i = 0
    name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_supported = False
    print(f'Device {i}: {name}')
    print('Total VRAM (GB):', round(props.total_memory / 1e9, 2))
    print('Compute capability:', props.major, props.minor)
    print('BF16 supported:', bf16_supported)
    print('FP16 supported:', True)
else:
    print('No CUDA device detected.')