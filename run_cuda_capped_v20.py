"""Run one GPU entry point with a fixed PyTorch allocator ceiling."""
import argparse
from pathlib import Path
import runpy
import sys
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap-gib', type=int, choices=(32, 48), required=True)
    parser.add_argument('script')
    args, rest = parser.parse_known_args()
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = args.cap_gib * 2**30 / total
    if not 0 < fraction <= 1:
        raise ValueError('allocator ceiling exceeds GPU capacity')
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    script = Path(args.script).resolve()
    print(f'v20_allocator_cap_gib={args.cap_gib} visible_gpu=0 script={script.name}', flush=True)
    sys.argv = [str(script), *rest]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
