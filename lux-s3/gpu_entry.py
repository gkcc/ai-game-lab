"""Initialize CUDA on the main thread, then invoke the unmodified official CLI."""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np


def main():
    devices = jax.devices()
    if not devices or not all(d.platform == "gpu" and "4080" in d.device_kind for d in devices):
        raise SystemExit(f"CUDA GPU is required; got {devices}")
    matrix = jnp.ones((32, 32), dtype=jnp.float32)
    result = jax.jit(lambda x: x @ x)(matrix)
    if not np.all(np.asarray(result) == 32):
        raise SystemExit("GPU warmup returned incorrect values")
    print(f"GPU warmup verified: {devices[0].device_kind}; loading official Lux CLI", file=sys.stderr, flush=True)
    from luxai_runner.cli import main as official_main
    official_main()


if __name__ == "__main__":
    main()
