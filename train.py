"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
python train.py --config-name=train_diffusion_lowdim_workspace \
    output_dir=/path/to/output
"""

# Inside the Isaac Sim Docker container PYTHONPATH puts Kit's bundled
# pip_prebundle (numpy < 2.0, numba 0.59) ahead of the conda env, so a plain
# ``import numpy`` here resolves to the Kit numpy and ``accelerate >= 1.0``'s
# ``np._core`` access blows up. Strip Isaac Sim's entries from sys.path AND
# evict any partially-cached numpy / numba modules from sys.modules (Python
# startup hooks may have already imported them from the Kit bundle), then
# re-import so the conda versions get cached. Local to this process — sibling
# Isaac Sim subprocesses spawned by the orchestrator have their own sys.path.
import sys
sys.path = [p for p in sys.path if "/isaac-sim/" not in p]
for _mod in list(sys.modules):
    if _mod == "numpy" or _mod.startswith("numpy.") or _mod == "numba" or _mod.startswith("numba."):
        del sys.modules[_mod]
import numpy  # noqa: E402, F401
import numba  # noqa: E402, F401

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra  # noqa: E402
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=cfg.output_dir)
    workspace.run()


if __name__ == "__main__":
    main()
