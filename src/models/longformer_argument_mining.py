import logging
from typing import Iterator, MutableMapping, Optional, Tuple, TypedDict, Union

import torch.nn
from pytorch_ie import Model
from pytorch_ie.models.common import ModelWithBoilerplate
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from torch import FloatTensor, LongTensor
from torch.nn import Parameter
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

# model inputs / outputs / targets
# InputType: TypeAlias = MutableMapping[str, LongTensor]
InputType: TypeAlias = MutableMapping[str, torch.Tensor]

OutputType: TypeAlias = TokenClassifierOutput
TargetType: TypeAlias = MutableMapping[str, torch.Tensor]

# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


logger = logging.getLogger(__name__)


@Model.register()
class AMParserModel(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
):
    def forward(self): ...
    def decode(self): ...
    def span_predict(self): ...
    def proposition_predict(self): ...
    def edge_predict(self): ...

    def save_model_file(self, model_file: str) -> None:
        return super().save_model_file(model_file)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        return super().load_model_file(model_file, map_location, strict)

    def load_original_checkpoint(self, model_file):
        # see load_model_file
        ...
