from typing import Tuple, Union

from torch import Tensor

from ..utils import logging
from .modeling_utils import Sideloads
from .transformer_2d import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
ModelOutputs = Union[Tensor, Transformer2DModelOutput]


class SideloadProcessor:
    """
    A sideload processor act as the mediator between source moduel(adaptor, control net) to its target layers(ex:
    unet.down_blocks.1) inside the target model. We can have one sideload process per model that shared across all
    layer, or even go as far as one processor per layer. This object also host the logic that fuse adapter state to
    model hidden state.
    """

    def __init__(self):
        self._sideloads = None

    def update_sideload(self, sideloads: Sideloads):
        """
        Loading new set of adapter states before main model which host all target layers
         begin new forward pass
        """
        if self._sideloads is not None and len(self._sideloads) > 0:
            unused_keys = list(self._sideloads.keys())
            logger.warning(f"Overriding unused sideload states: {unused_keys}")

        self._sideloads = sideloads.clone()

    def merge_states(self, module_state: ModelOutputs, sideload_state: Tensor) -> ModelOutputs:
        """
        Parameters:
            module_state: hidden state/output of targeted layer(nn.module)
            sideload_state: one of the hidden state outputed from adapter
        """
        if isinstance(module_state, Tensor):
            return module_state + sideload_state
        elif isinstance(module_state, Transformer2DModelOutput):
            sample = module_state.sample + sideload_state
            return Transformer2DModelOutput(sample=sample)
        else:
            raise ValueError(f"SideloadProcessor got a unsupported data type: {type(module_state)}")

    def __call__(self, module_name: str, module_output: Union[ModelOutputs, Tuple[ModelOutputs]]):
        """
        Lookup and merge the adapter state correspond to a specific module. Note that all the adapter state stored in
        the processor are one time use, adapter state will be remove after being merge.

        Parameters:
            module_name: name of the caller(a child nn.module of the main model)
            module_output: Tensor or dataclass outputed by target layer
        """
        if self._sideloads and module_name in self._sideloads:
            sideload = self._sideloads.pop(module_name)

            if isinstance(module_output, tuple):
                new_output = tuple(self.merge_states(m, s) for m, s in zip(module_output, sideload))
            else:
                new_output = self.merge_states(module_output, sideload)
            return new_output
        else:
            return module_output
