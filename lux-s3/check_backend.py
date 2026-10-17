"""Small synchronized JAX and official environment checks; no training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "backend.json")
    parser.add_argument("--batch", type=int, choices=(1, 4, 8, 16, 32, 64), default=4)
    parser.add_argument("--steps", type=int, choices=(32, 64, 128, 256), default=32)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    def progress(stage):
        if args.progress:
            print(stage, file=sys.stderr, flush=True)
    devices = jax.devices()
    progress(f"JAX devices: {devices}")
    if args.require_gpu and not any(d.platform == "gpu" for d in devices):
        raise SystemExit("GPU required, but JAX has no GPU backend; CPU fallback rejected")
    gpu_backend = jax.default_backend() == "gpu"
    def synchronize(value):
        if gpu_backend:
            # On this WSL/legacy-JAX combination GetReadyFuture can stall in
            # cuStreamCreate. A blocking host read verifies completed results
            # through the same transfer path used by the official CLI.
            for leaf in jax.tree_util.tree_leaves(value):
                if isinstance(leaf, jax.Array):
                    np.asarray(leaf)
        else:
            jax.block_until_ready(value)
        return value
    x = jnp.ones((256, 256), dtype=jnp.float32)
    multiply = jax.jit(lambda a: a @ a)
    synchronize(multiply(x))
    progress("Matrix warmup complete")
    started = time.perf_counter()
    product = synchronize(multiply(x))
    matmul_seconds = time.perf_counter() - started
    assert np.all(np.asarray(product) == 256)
    progress("Matrix calculation and host result verified")

    from luxai_s3.env import LuxAIS3Env
    from luxai_s3.params import EnvParams
    progress("Official environment modules loaded")
    env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
    params = env.default_params
    batch = args.batch
    steps = args.steps
    keys = jax.random.split(jax.random.key(42), batch)
    reset = jax.jit(jax.vmap(lambda key: env.reset(key, params)))
    obs, state = reset(keys)
    synchronize(state)
    progress("Official environment reset complete")
    actions = {f"player_{i}": jnp.zeros((params.max_units, 3), dtype=jnp.int32) for i in range(params.num_teams)}
    step = jax.jit(jax.vmap(lambda key, state: env.step(key, state, actions, params)))
    _, state, *_ = step(keys, state)
    synchronize(state)
    progress("Official environment step warmup complete")
    started = time.perf_counter()
    for _ in range(steps):
        _, state, *_ = step(keys, state)
    synchronize(state)
    duration = time.perf_counter() - started
    assert np.all(np.asarray(state.steps) == steps + 1)
    state_devices = sorted({str(d) for leaf in jax.tree_util.tree_leaves(state)
                            if isinstance(leaf, jax.Array) for d in leaf.devices()})
    actual_gpu = (all(d.platform == "gpu" for d in product.devices())
                  and all(d.platform == "gpu"
                          for leaf in jax.tree_util.tree_leaves(state) if isinstance(leaf, jax.Array)
                          for d in leaf.devices()))
    if args.require_gpu and not actual_gpu:
        raise SystemExit("Matrix and official environment state must both execute on CUDA GPU")
    stats = {"os": platform.platform(), "python": platform.python_version(),
             "jax": jax.__version__, "jaxlib": importlib.metadata.version("jaxlib"),
             "backend": jax.default_backend(),
             "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in devices],
             "array_devices": [str(d) for d in product.devices()],
             "environment_state_devices": state_devices,
             "matmul_shape": [256, 256], "matmul_verified": True, "matmul_seconds": matmul_seconds,
             "official_env_batch": batch, "timed_steps_per_env": steps,
             "timed_total_env_steps": batch * steps, "environment_seconds": duration,
             "environment_steps_per_second": batch * steps / duration,
             "environment_state_steps": np.asarray(state.steps).tolist(),
             "timing": "JIT warmup excluded; synchronized before and after timing",
             "synchronization": "blocking host read of every result array; final transfer cost included" if gpu_backend else "block_until_ready",
             "scope": f"{batch} environments with zero actions; excludes reset, Bot I/O and replay serialization. Small sample, not full-game throughput.",
             "cuda_gpu_actually_used": actual_gpu,
             "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])}
    if sys.platform == "win32":
        stats["wsl_distros"] = subprocess.run(["wsl", "--list", "--verbose"], capture_output=True).stdout.decode("utf-16-le", errors="replace")
        stats["gpu_blocker"] = "This native Windows virtual environment is CPU-only; use the isolated WSL2 GPU environment"
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
