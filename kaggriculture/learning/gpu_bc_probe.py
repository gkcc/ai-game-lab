"""Small real-teacher BC feasibility/profiling on RTX and CPU, not a match Bot."""
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

out = Path(sys.argv[1]).resolve()
data = np.load(out / "dataset.npz")
features, labels, validation = data["x"], data["y"], data["validation"]
x, y = features[~validation], labels[~validation]
rng = np.random.default_rng(20261018)
initial = (rng.normal(0,.05,(x.shape[1],128)).astype(np.float32), np.zeros(128,np.float32),
           rng.normal(0,.05,(128,5)).astype(np.float32), np.zeros(5,np.float32))
indices = rng.integers(0,len(y),size=(512,256),dtype=np.int32)
def logits(w, x):
    return jax.nn.relu(x@w[0]+w[1])@w[2]+w[3]
def loss(w, x, y):
    logp = jax.nn.log_softmax(logits(w,x))
    return -logp[jnp.arange(y.shape[0]), y].mean()
def fit(w,x,y,indices):
    def step(weights, ids):
        value, grads = jax.value_and_grad(loss)(weights,x[ids],y[ids])
        return jax.tree_util.tree_map(lambda a,g:a-.03*g,weights,grads),value
    return jax.lax.scan(step,w,indices)
records = {}
weights_host = {}
for backend in ("cpu","gpu"):
    device = jax.devices(backend)[0]
    with jax.default_device(device):
        w = jax.device_put(initial,device)
        inputs = [jax.device_put(a,device) for a in (x,y,indices)]
        compiled = jax.jit(fit)
        tick = time.perf_counter()
        warm, losses = compiled(w,*inputs)
        # WSL GetReadyFuture workaround: blocking host copy of EVERY output.
        [np.asarray(a) for a in jax.tree_util.tree_leaves((warm,losses))]
        compile_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        learned, losses = compiled(w,*inputs)
        host = [np.asarray(a) for a in learned]
        losses = np.asarray(losses)
        elapsed = time.perf_counter()-tick
        weights_host[backend] = host
        def accuracy(xx,yy):
            probs = np.maximum(0,xx@host[0]+host[1])@host[2]+host[3]
            return float((probs.argmax(1)==yy).mean())
        records[backend] = {"device": str(device), "device_kind": device.device_kind,
                            "parameter_devices": [str(a.devices()) for a in learned],
                            "compile_and_warm_seconds": compile_seconds,"timed_seconds": elapsed,
                            "gradient_updates":512,"batch_size":256,"updates_per_second":512/elapsed,
                            "initial_minibatch_loss":float(losses[0]),"final_minibatch_loss":float(losses[-1]),
                            "train_accuracy":accuracy(x,y),
                            "validation_accuracy":accuracy(features[validation],labels[validation]),
                            "weights_changed":any(not np.array_equal(a,b) for a,b in zip(initial,host))}
        np.savez_compressed(out / f"bc-crop-{backend}.npz",**{f"w{i}":a for i,a in enumerate(host)})
assert records["gpu"]["parameter_devices"]
assert all(r["weights_changed"] and r["final_minibatch_loss"] < r["initial_minibatch_loss"] for r in records.values())
result = {"passed": True,"jax":jax.__version__,"records":records,
          "gpu_speedup_excluding_compile":records["cpu"]["timed_seconds"]/records["gpu"]["timed_seconds"],
          "max_cpu_gpu_weight_difference":max(float(np.abs(a-b).max()) for a,b in zip(weights_host['cpu'],weights_host['gpu'])),
          "synchronization":"blocking host copies of all updated weights and all losses, transfer time included",
          "scope":"Actual gradient training on official v1 match PLANT labels. Crop classifier only, not integrated as an agent; no claim of improved play or accelerated environment"}
(out / "gpu-bc-probe.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
