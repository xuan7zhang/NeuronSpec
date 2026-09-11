import os, sys, time
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2"); os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ["NEURON_COMPILE_CACHE_URL"] = "/scratch/neuron-cache/train"; os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --lnc=2"
sys.path.insert(0, "/home/ec2-user/mismatch"); import recompute as R
import torch, torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16); print("params %.2fB" % (sum(p.numel() for p in m.parameters())/1e9), flush=True)
dev = xm.xla_device(); m = m.to(dev).eval()
ids = torch.randint(0, 1000, (1, 1024)).to(dev); attn = torch.ones(1, 1024, dtype=torch.long).to(dev)
t0 = time.time()
with torch.no_grad():
    lp = R.forward_scalar(m, ids, attn); xm.mark_step(); lp = lp.cpu()
print(f"PROBE_OK 8B B=1 T=1024 lp mean {lp.float().mean().item():.3f} elapsed {time.time()-t0:.0f}s", flush=True)
