"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import logging
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import torch
from pie_core import Annotation, AnnotationLayer, Document, TaskEncoding, TaskModule
from pie_core.metric import EncodingMetric
from pie_documents.annotations import (
    BinaryRelation,
    LabeledSpan,
    MultiLabeledBinaryRelation,
    NaryRelation,
    Span,
)
from pie_documents.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TokenBasedDocument,
    TokenDocumentWithLabeledSpansAndBinaryRelations,
    TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pie_documents.utils.span import distance as span_distance
from pie_documents.utils.span import is_contained_in
from pytorch_ie.models.simple_sequence_classification import InputType as ModelInputType
from pytorch_ie.models.simple_sequence_classification import TargetType as ModelTargetType
from pytorch_ie.taskmodules.common.mixins import RelationStatisticsMixin
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from pytorch_ie.utils.document import tokenize_document
from pytorch_ie.utils.window import get_window_around_slice
from torch import FloatTensor, LongTensor
from torchmetrics import ClasswiseWrapper, F1Score, Metric, MetricCollection
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias, TypeVar

InputEncodingType: TypeAlias = Dict[str, Any]
TargetEncodingType: TypeAlias = Sequence[int]
DocumentType: TypeAlias = TextBasedDocument

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    labels: Sequence[str]
    probabilities: Sequence[float]


TaskModuleType: TypeAlias = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    Tuple[ModelInputType, Optional[ModelTargetType]],
    ModelTargetType,
    TaskOutputType,
]


HEAD = "head"
TAIL = "tail"
START = "start"
END = "end"


logger = logging.getLogger(__name__)


@TaskModule.register()
class AMTaskModule(TaskModuleType, RelationStatisticsMixin):
    """TODO: Change everything here
    Marker based relation extraction. This taskmodule prepares the input token ids in such a way
    that before and after the candidate head and tail entities special marker tokens are inserted.
    Then, the modified token ids can be simply passed into a transformer based text classifier
    model.

    parameters:
        abc: ???
    """

    # PREPARED_ATTRIBUTES = ["labels", "entity_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        pass

    def _post_prepare(self):
        pass

    def encode(self, *args, **kwargs):
        self.reset_statistics()
        res = super().encode(*args, **kwargs)
        self.show_statistics()
        return res

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        tokenized_document_type = TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions

        casted_document_type = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        casted_document = document.as_type(casted_document_type)

        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=tokenized_document_type,
        )
        return tokenized_docs

    def encode_target(self, task_encoding: TaskEncoding[TextBasedDocument, Dict[str, Any], Sequence[int]]) -> Sequence[int] | None:  # type: ignore[empty-body]
        pass

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:  # type: ignore[empty-body]
        pass

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> Tuple[ModelInputType]:  # type: ignore[empty-body]
        pass

    def create_annotations_from_output(self, task_encoding: TaskEncodingType, task_output: TaskOutputType) -> Iterator[Tuple[str, Annotation]]:  # type: ignore[empty-body]
        pass

    def configure_model_metric(self, stage: str) -> Metric | MetricCollection | None:  # type: ignore[empty-body]
        return None
