"""实验用 import hook, 通过 PYTHONPATH=hooks 注入 (不改 venv), env 门控, 默认惰性无副作用:
MISMATCH_CTE_OPT=<N> 时, 在 neuronx_distributed_inference.models.model_wrapper 被 import 后
把 ModelWrapper 对 context-encoding 图硬编码的 -O1 换成 -O<N>, 并去掉强制 modular-flow 的阈值。
只改 CTE 图的编译等级, TKG 图不变。不在解释器启动时 import 任何 Neuron 模块。"""
import importlib.abc, importlib.util, os, sys

TARGET = "neuronx_distributed_inference.models.model_wrapper"

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        self._orig_loader = spec.loader
        spec.loader = self
        return spec
    def create_module(self, spec):
        return self._orig_loader.create_module(spec)
    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        lvl = os.environ.get("MISMATCH_CTE_OPT")
        orig = module.ModelWrapper.__init__
        def __init__(self, *a, **k):
            orig(self, *a, **k)
            if self.tag == module.CONTEXT_ENCODING_MODEL_TAG and " -O1 " in self.compiler_args:
                self.compiler_args = self.compiler_args.replace(" -O1 ", f" -O{lvl} ").replace(" --modular-flow-mac-threshold=10 ", " ")
                print(f"[mismatch_cte_hook] CTE compiler_args -> {self.compiler_args}", flush=True)
        module.ModelWrapper.__init__ = __init__
        print(f"[mismatch_cte_hook] armed: CTE -O1 -> -O{lvl}", flush=True)

if os.environ.get("MISMATCH_CTE_OPT"):
    sys.meta_path.insert(0, _Finder())
