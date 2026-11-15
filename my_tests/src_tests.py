from functools import partial
import os
from hydra_zen import instantiate, just, make_custom_builds_fn
from hydra_zen.third_party.pydantic import validates_with_pydantic
from hydra_zen.third_party.beartype import validates_with_beartype
from beartype.cave import ScalarTypes
from torchvision.models import resnet18, resnet50

@validates_with_beartype
class Optimizer:
    def __init__(self, model_weights, learning_rate):
        self.weights = model_weights
        self.lr = learning_rate

def logger(message: str, format_spec: str) -> str:
    return format_spec.format(message)

def try_hydra_partials():
    builds = make_custom_builds_fn(
        populate_full_signature=True,
    )
    pbuilds = make_custom_builds_fn(
        zen_partial=True, 
        populate_full_signature=True,
        
        # zen_wrappers=validates_with_beartype
    )

    OptimConf = builds(Optimizer, learning_rate=0.1, model_weights=[1, 2, 3])
    print("Full build of optim conf", OptimConf)
    built_optim = instantiate(OptimConf)
    print("Built optimiser from full: ", built_optim)
    print("Attribute of built optimiser", built_optim.lr)

    OptimConf = builds(Optimizer, learning_rate=0.1, model_weights=[1, 2, 3], zen_partial=True)
    print("(Intended) Full build of optim conf", OptimConf)
    built_optim = instantiate(OptimConf)
    print("Built optimiser from (intended) full: ", built_optim)
    try:
        print("Attribute of built optimiser", built_optim.lr) # type: ignore
    except:
        print("Attribute of built optimiser", built_optim().lr)

    pOptimConf = pbuilds(Optimizer)
    print("Partial build of optim conf: ", pOptimConf)
    # print(pOptimConf(learning_rate=0.1, model_weights=[1,2,3]))
    # print(instantiate(pOptimConf)(learning_rate=0.1, model_weights=[1,2,3]))
    pOptimFuncToolsObj = instantiate(pOptimConf)
    print("Instantiated object: partial configuration: ", pOptimFuncToolsObj)
    
    built_optim = pOptimFuncToolsObj(learning_rate=0.1, model_weights=[1, 2, 3])
    print("Built optimiser from partial: ", built_optim)
    print("Attribute of built optimiser", built_optim.lr)

    print("Finish partial testing for optim")

    def f(x: ScalarTypes): return x  # type: ignore # a scalar is any real-valued number
    f([1, 2])
    val_f = validates_with_beartype(f)
    # val_f([1, 2])

    class A:
        def __init__(self, x: ScalarTypes): ... # type: ignore
    validates_with_beartype(A)  # wrapping occurs in-place
    # A([1, 2])

    partial_logger = partial(logger, format_spec='{0:>8s}')
    LogConf = just(partial_logger)
    print(LogConf)
    # partiald_optim = instantiate(OptimConf)
    # print(partiald_optim)

    # partiald_optim(model_weights=[1, 2, 3])

    # partiald_logger = instantiate(LogConf)
    # print(partiald_logger)

    # partiald_logger("hello")
    # print(partiald_logger("goodbye")) 

def sample_resnet_params():
    model50 = resnet50()
    print(model50.fc)
    print(model50.conv1)
    model18 = resnet18()
    print(model18.fc)
    print(model18.conv1)

def get_cuda_capabilities():
    print(f"CUDA JIT Force Level: {os.environ.get('CUDA_MODULE_LOADING', 'Default')}")

if __name__ == "__main__":
    try_hydra_partials()