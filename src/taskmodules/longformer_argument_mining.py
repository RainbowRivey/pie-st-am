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
from copy import deepcopy
from dataclasses import dataclass
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
from pytorch_ie.taskmodules.common.mixins import RelationStatisticsMixin
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from pytorch_ie.utils.document import tokenize_document
from pytorch_ie.utils.window import get_window_around_slice
from tokenizers import Encoding
from torch import FloatTensor, LongTensor
from torchmetrics import ClasswiseWrapper, F1Score, Metric, MetricCollection
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias, TypeVar

from src.models.longformer_argument_mining import InputType as ModelInputType
from src.models.longformer_argument_mining import TargetType as ModelTargetType


@dataclass
class InputEncodingType:
    input_ids: Sequence[int]
    attention_mask: Sequence[int]
    span_start: Optional[Sequence[int]] = None
    span_end: Optional[Sequence[int]] = None
    span_label: Optional[Sequence[int]] = None


@dataclass
class TargetEncodingType:
    rel_head: Sequence[int]
    rel_tail: Sequence[int]
    rel_label: Sequence[int]

    span_start: Optional[Sequence[int]] = None
    span_end: Optional[Sequence[int]] = None
    span_label: Optional[Sequence[int]] = None


DocumentType: TypeAlias = TextDocumentWithLabeledSpansAndBinaryRelations

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


@dataclass
class TaskOutputType:
    span_start: Sequence[int]
    span_end: Sequence[int]
    span_label: Sequence[int]

    rel_head: Sequence[int]
    rel_tail: Sequence[int]
    rel_label: Sequence[int]

    span_probability: Sequence[float] | None = None
    span_label_probability: Sequence[float] | None = None

    rel_probability: Sequence[float] | None = None
    rel_label_probability: Sequence[float] | None = None


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
class SpansAndBinaryRelationsTaskModule(TaskModuleType, RelationStatisticsMixin):
    """TaskModule for End to End Argument Mining."""

    PREPARED_ATTRIBUTES = ["relation_labels", "entity_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        none_label: str = "no_relation",
        partition_annotation: Optional[str] = None,
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        relation_labels: Optional[List[str]] = None,
        entity_labels: Optional[List[str]] = None,
        use_gold_spans: bool = False,
        **kwargs,
    ) -> None:
        """

        Args:
            tokenizer_name_or_path (str): _description_
            partition_annotation (Optional[str], optional): _description_. Defaults to None.
            tokenize_kwargs (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
            relation_labels (Optional[List[str]], optional): _description_. Defaults to None.
            entity_labels (Optional[List[str]], optional): _description_. Defaults to None.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.partition_annotation = partition_annotation
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.tokenize_kwargs = tokenize_kwargs or {}

        self.relation_labels = relation_labels
        self.entity_labels = entity_labels
        self.none_label = none_label
        self.use_gold_spans = use_gold_spans

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        entity_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for document in documents:
            relations: AnnotationLayer[BinaryRelation] = document.binary_relations
            entities: AnnotationLayer[LabeledSpan] = document.labeled_spans

            for entity in entities:
                entity_labels.add(entity.label)

            for relation in relations:
                relation_labels.add(relation.label)

        if self.none_label in relation_labels:
            relation_labels.remove(self.none_label)

        self.label_to_id = {label: i + 1 for i, label in enumerate(sorted(relation_labels))}
        self.label_to_id[self.none_label] = 0

        self.entity_labels = sorted(entity_labels)
        self.relation_labels = sorted(relation_labels)

    def _post_prepare(self):
        if self.entity_labels is None or self.relation_labels is None:
            raise ValueError("Entity labels and relation labels must be set after preparation.")

        self.entity_label_to_id = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.relation_label_to_id = {label: idx for idx, label in enumerate(self.relation_labels)}

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
        if self.partition_annotation is None:
            casted_document.labeled_partitions.append(
                LabeledSpan(0, len(casted_document.text), "text")
            )

        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=tokenized_document_type,
            partition_layer="labeled_partitions",
            add_special_tokens=True,
            strict_span_conversion=False,
            return_overflowing_tokens=True,
            **self.tokenize_kwargs,
        )

        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc in tokenized_docs:
            inputs = tokenized_doc.metadata["tokenizer_encoding"]
            metadata: dict = {"tokenized_document": tokenized_doc}
            if self.use_gold_spans:
                encoded_spans, span_to_id = self.encode_spans(tokenized_doc)
                # inputs.update(encoded_spans)
                for k, v in encoded_spans.items():
                    setattr(inputs, k, v)
                metadata["span_to_id"] = span_to_id

            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=inputs,
                    metadata=metadata,
                )
            )

        return task_encodings

    def encode_target(self, task_encoding: TaskEncodingType) -> Optional[TargetEncodingType]:
        metadata = task_encoding.metadata
        tokenized_document = metadata["tokenized_document"]

        targets: dict[str, list[int]] = {
            "rel_head": [],
            "rel_tail": [],
            "rel_label": [],
        }

        encoded_spans, span_to_id = self.encode_spans(tokenized_document)
        targets.update(encoded_spans)
        task_encoding.metadata["span_to_id"] = span_to_id

        for binary_relation in tokenized_document.binary_relations:
            targets["rel_head"].append(span_to_id[binary_relation.head])
            targets["rel_tail"].append(span_to_id[binary_relation.tail])
            targets["rel_label"].append(self.relation_label_to_id[binary_relation.label])

        return TargetEncodingType(**targets)

    def encode_spans(
        self,
        tokenized_document: TokenDocumentWithLabeledSpansAndBinaryRelations,
    ):

        span_to_id = {}
        result = defaultdict(list)
        for span_id, labeled_span in enumerate(tokenized_document.labeled_spans):
            result["span_start"].append(labeled_span.start)
            result["span_end"].append(labeled_span.end)
            result["span_label"].append(self.entity_label_to_id[labeled_span.label])
            span_to_id[labeled_span] = span_id

        return result, span_to_id

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        unbatched_output = []
        outputs = {}
        for key, value in model_output.items():
            outputs[key] = value.detach().cpu().tolist()

        for batch_idx in range(len(outputs["rel_label"])):
            result = {}
            for key, value in outputs.items():
                result[key] = [v for v in value[batch_idx] if v >= 0]

            unbatched_output.append(TaskOutputType(**result))

        return unbatched_output

    def collate(
        self, task_encodings: Sequence[TaskEncodingType]
    ) -> Tuple[ModelInputType, ModelTargetType | None]:
        input_ids_list = []
        attention_mask_list = []

        rel_lists = defaultdict(list)
        span_lists = defaultdict(list)

        construct_targets = any(encoding.has_targets for encoding in task_encodings)

        for task_encoding in task_encodings:
            tokenizer_encoding: Encoding = task_encoding.metadata["tokenized_document"].metadata[
                "tokenizer_encoding"
            ]
            input_ids_list.append(tokenizer_encoding.ids)
            attention_mask_list.append(tokenizer_encoding.attention_mask)
            # if self.use_gold_spans:
            # span_lists["span_start"].append(task_encoding.inputs.span_start)
            # span_lists["span_end"].append(task_encoding.inputs.span_end)
            # span_lists["span_label"].append(task_encoding.inputs.span_label)
            if task_encoding.has_targets:
                enc_targets = task_encoding.targets
                # if not self.use_gold_spans:
                span_lists["span_start"].append(enc_targets.span_start)
                span_lists["span_end"].append(enc_targets.span_end)
                span_lists["span_label"].append(enc_targets.span_label)
                rel_lists["rel_head"].append(enc_targets.rel_head)
                rel_lists["rel_tail"].append(enc_targets.rel_tail)
                rel_lists["rel_label"].append(enc_targets.rel_label)

        targets = None

        # if self.use_gold_spans or construct_targets:
        padded_span_encodings = self.pad(
            span_lists,
            pad_ids={k: -1 for k in span_lists},
            return_tensors=True,
        )

        padded_model_input = self.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            },
            return_tensors="pt",
            padding=True,
            max_length=self.tokenize_kwargs.get("max_length", None),
        )
        model_input: dict[str, torch.Tensor] = {
            "input_ids": padded_model_input["input_ids"],
            "attention_mask": padded_model_input["attention_mask"],
        }

        # if self.use_gold_spans:
        model_input.update(padded_span_encodings)

        if construct_targets:
            targets = self.pad(rel_lists, pad_ids={k: -1 for k in rel_lists}, return_tensors=True)

            # if not self.use_gold_spans:
            targets.update(padded_span_encodings)

        return (model_input, targets)

    def decode_annotations(
        self,
        encoding: TaskOutputType,
        tokenized_document: TokenDocumentWithLabeledSpansAndBinaryRelations,
    ) -> dict[str, List[Annotation]]:
        entities: list[Annotation] = []
        relations: list[Annotation] = []
        tokenizer_encoding: Encoding = tokenized_document.metadata["tokenizer_encoding"]
        token_offset_mapping = tokenized_document.metadata["token_offset_mapping"]

        if self.relation_labels is None:
            raise ValueError(
                "'relation_labels' must be set before calling decode_annotations(). Was prepare() called on the taskmodule?"
            )
        if self.entity_labels is None:
            raise ValueError(
                "'entity_labels' must be set before calling decode_annotations(). Was prepare() called on the taskmodule?"
            )

        for span_start, span_end, span_probability, span_label, span_label_probability in zip(
            encoding.span_start,
            # Span start probability
            encoding.span_end,
            # Span end probability
            getattr(encoding, "span_probability", [1.0] * len(encoding.span_start)),
            encoding.span_label,
            getattr(encoding, "span_label_probability", [1.0] * len(encoding.span_label)),
        ):
            if span_end < 1:
                raise ValueError("Decoded span_end is less than 1, invalid span.")
            entity = LabeledSpan(
                start=token_offset_mapping[span_start][0],
                end=token_offset_mapping[span_end - 1][1],
                label=self.entity_labels[span_label],
                score=span_probability * span_label_probability,
            )
            entities.append(entity)

        for (
            head_id,
            tail_id,
            relation_probability,
            relation_label,
            relation_label_probability,
        ) in zip(
            encoding.rel_head,
            encoding.rel_tail,
            getattr(encoding, "rel_probability", [1.0] * len(encoding.rel_head)),
            encoding.rel_label,
            getattr(encoding, "rel_label_probability", [1.0] * len(encoding.rel_label)),
        ):
            relation = BinaryRelation(
                head=entities[head_id],
                tail=entities[tail_id],
                label=self.relation_labels[relation_label],
                score=relation_probability * relation_label_probability,
            )
            relations.append(relation)

        return {"labeled_spans": entities, "binary_relations": relations}

    def create_annotations_from_output(
        self, task_encoding: TaskEncodingType, task_output: TaskOutputType
    ) -> Iterator[Tuple[str, Annotation]]:
        tokenized_document = task_encoding.metadata["tokenized_document"]
        annotations = self.decode_annotations(task_output, tokenized_document)
        for ann_type, anns in annotations.items():
            for ann in anns:
                yield ann_type, ann

    def pad(
        self,
        encoded_input: Dict[str, List[List[int]]],
        pad_ids: Dict[str, int],
        return_tensors: bool = False,
    ) -> dict[str, Any]:
        """Pad encoded inputs (on right and up to max length in the batch)"""
        result = deepcopy(encoded_input)
        for key, pad_id in pad_ids.items():
            required_input = result[key]
            max_length = max(len(_input) for _input in required_input)

            for i in range(len(required_input)):
                difference = max_length - len(required_input[i])
                result[key][i] = required_input[i] + [pad_id] * difference

        if return_tensors:
            return {k: torch.tensor(v) if k in pad_ids else v for k, v in result.items()}

        return result

    def configure_model_metric(self, stage: str) -> Metric | MetricCollection | None:  # type: ignore[empty-body]
        return None
