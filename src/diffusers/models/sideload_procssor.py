from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor
from .transformer_2d import Transformer2DModelOutput
from .modeling_utils import Sideloads

ModelOutputs = Union[Tensor, Transformer2DModelOutput]


class SideloadProcessor:

    def __init__(self):
        self._sideloads = None
    
    def update_sideload(self, sideloads: Sideloads):
        self._sideloads = sideloads.clone()
    
    def merge_states(self, module_state: ModelOutputs, sideload_state: Tensor) -> ModelOutputs:
        if isinstance(module_state, Tensor):
            return module_state + sideload_state
        elif isinstance(module_state, Transformer2DModelOutput):
            sample = module_state.sample + sideload_state
            return Transformer2DModelOutput(sample=sample)
        else:
            raise ValueError(f"SideloadProcessor got a unsupported data type: {type(module_state)}")
    
    def __call__(self, module_name: str, module_output: Union[ModelOutputs, Tuple[ModelOutputs]]):
        if self._sideloads and module_name in self._sideloads:
            sideload = self._sideloads[module_name]
            contained = isinstance(module_output, tuple)
            
            sideload = sideload if contained else (sideload,)
            module_output = module_output if contained else (module_output,)
            
            new_output = tuple(self.merge_states(m, s) for s, m in zip(sideload, module_output))
            # print('SideloadProcessor', module_name, sideload[0].shape)
            del self._sideloads[module_name]
            return new_output if contained else new_output[0]
        else:
            return module_output