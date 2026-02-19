import json
import logging
from dataclasses import asdict, replace

import numpy as np
import pytest
from pie_datasets import load_dataset
from pie_documents.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.documents import TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
from torch import allclose, load, save, tensor

from src.taskmodules import AMTaskModule
from src.taskmodules.longformer_argument_mining import (
    ModelTargetType,
    TargetEncodingType,
    TaskEncodingType,
    TaskOutputType,
)
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def documents():
    dataset = load_dataset("pie/cdcp").to_document_type(
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    )
    documents = dataset["test"][:3]
    return documents


@pytest.fixture(scope="module")
def document(documents):
    return documents[0]


@pytest.fixture(scope="module")
def taskmodule(documents):
    taskmodule = AMTaskModule(tokenizer_name_or_path="allenai/longformer-base-4096")
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture(scope="module")
def target_encoding():
    return TargetEncodingType(
        **{
            "rel_head": [2],
            "rel_tail": [1],
            "rel_label": [0],
            "span_start": [1, 26, 41],
            "span_end": [26, 41, 76],
            "span_label": [0, 2, 1],
        }
    )


@pytest.fixture(scope="module")
def task_output(target_encoding):
    task_output = asdict(target_encoding)
    task_output.update(
        {
            "span_probability": [0.9, 0.85, 0.8],
            "span_label_probability": [0.95, 0.9, 0.85],
            "rel_probability": [0.88],
            "rel_label_probability": [0.93],
        }
    )

    return TaskOutputType(**task_output)


@pytest.fixture(scope="module")
def input_encoding(document, taskmodule):
    encoded_inputs = taskmodule.encode_input(document)
    return encoded_inputs[0]


@pytest.fixture(scope="module")
def task_encoding(document, taskmodule):
    return taskmodule.encode_input(document)[0]


def test_pad(taskmodule):
    padded = taskmodule.pad(
        {
            "input_ids": [[0, 1, 2], [0, 1]],
            "span_start": [[1, 26, 41], [2, 30]],
            "span_end": [[26, 41, 76], [30, 50]],
            "span_label": [[0, 2, 1], [1, 0]],
        },
        pad_ids={"input_ids": -1, "span_start": -1, "span_end": -1, "span_label": -1},
        return_tensors=True,
    )

    expected = {
        "input_ids": tensor([[0, 1, 2], [0, 1, -1]]),
        "span_start": tensor([[1, 26, 41], [2, 30, -1]]),
        "span_end": tensor([[26, 41, 76], [30, 50, -1]]),
        "span_label": tensor([[0, 2, 1], [1, 0, -1]]),
    }

    for k in padded:
        assert allclose(padded[k], expected[k])


def test_encode_input(taskmodule, document):
    encoded_input = taskmodule.encode_input(document)
    assert len(encoded_input) == 1
    assert (
        len(encoded_input[0].inputs.ids)
        == len(encoded_input[0].inputs.tokens)
        == len(encoded_input[0].inputs.offsets)
    )

    # Original outdated method used spaces for tokenization. Latest tokenizers use 'Ġ' to indicate spaces.
    # fmt: off
    assert encoded_input[0].inputs.tokens == [
        '<s>', 'Recently', ',', 'Ġcourts', 'Ġhave', 'Ġheld', 'Ġthat', 'Ġdebt', 'Ġcollectors', 'Ġcan', 'Ġescape', 'Ġ16', '92', 'i', "'s", 'Ġvenue', 'Ġprovisions', 'Ġentirely', 'Ġby', 'Ġpursuing', 'Ġdebt', 'Ġcollection', 'Ġthrough', 'Ġarbitration', 'Ġinstead', '.', 'ĠAs', 'Ġthe', 'ĠN', 'AF', 'Ġstudies', 'Ġreflect', ',', 'Ġarbitration', 'Ġhas', 'Ġnot', 'Ġproven', 'Ġa', 'Ġsatisfactory', 'Ġalternative', '.', 'ĠI', 'Ġurge', 'Ġthe', 'ĠC', 'FP', 'B', 'Ġto', 'Ġinclude', 'Ġin', 'Ġa', 'Ġrule', 'Ġlanguage', 'Ġinterpreting', 'Ġ16', '92', 'i', 'Ġas', 'Ġrequiring', 'Ġdebt', 'Ġcollectors', 'Ġto', 'Ġproceed', 'Ġin', 'Ġcourt', ',', 'Ġnot', 'Ġthrough', 'Ġlargely', '-', 'un', 'regulated', 'Ġarbit', 'ral', 'Ġforums', '.', '</s>'
        ]
    assert encoded_input[0].inputs.ids == [
            0, 38386,     6,  4354,    33,   547,    14,  1126, 24122,    64,
         5111,   545,  6617,   118,    18,  5584,  7668,  4378,    30,  8592,
         1126,  2783,   149, 16211,  1386,     4,   287,     5,   234,  8573,
         3218,  4227,     6, 16211,    34,    45,  5401,    10, 28173,  3626,
            4,    38,  8745,     5,   230,  9763,   387,     7,   680,    11,
           10,  2178,  2777, 37212,   545,  6617,   118,    25,  7980,  1126,
        24122,     7,  9073,    11,   461,     6,    45,   149,  2743,    12,
          879, 35908, 21552,  7085, 20678,     4,     2
    ]
    # Originally they sent offsets without cls and sep tokens to model
    # TODO: Check if this is still needed
    assert encoded_input[0].inputs.offsets == [
        (0, 0), (0, 8), (8, 9), (10, 16), (17, 21), (22, 26), (27, 31), (32, 36), (37, 47), (48, 51), (52, 58), (59, 61), (61, 63), (63, 64), (64, 66), (67, 72), (73, 83), (84, 92), (93, 95), (96, 104), (105, 109), (110, 120), (121, 128), (129, 140), (141, 148), (148, 149), (150, 152), (153, 156), (157, 158), (158, 160), (161, 168), (169, 176), (176, 177), (178, 189), (190, 193), (194, 197), (198, 204), (205, 206), (207, 219), (220, 231), (231, 232), (233, 234), (235, 239), (240, 243), (244, 245), (245, 247), (247, 248), (249, 251), (252, 259), (260, 262), (263, 264), (265, 269), (270, 278), (279, 291), (292, 294), (294, 296), (296, 297), (298, 300), (301, 310), (311, 315), (316, 326), (327, 329), (330, 337), (338, 340), (341, 346), (346, 347), (348, 351), (352, 359), (360, 367), (367, 368), (368, 370), (370, 379), (380, 385), (385, 388), (389, 395), (395, 396), (0, 0)
    ]
    # fmt: on


# encode_target is called by encode to get all targets
def test_encode(taskmodule, document):
    encoded_doc = taskmodule.encode(document, encode_target=True)
    assert len(encoded_doc) == 1
    target = encoded_doc[0].targets
    assert asdict(target) == {
        "span_start": [1, 26, 41],
        "span_end": [26, 41, 76],
        "span_label": [0, 2, 1],
        "rel_head": [2],
        "rel_tail": [1],
        "rel_label": [0],
    }


def test_create_annotations_from_output(taskmodule, task_output, task_encoding):
    document = task_encoding.document.copy()
    for annotation_name, annotation in taskmodule.create_annotations_from_output(
        task_encoding, task_output
    ):
        document[annotation_name].predictions.append(annotation)

    entities = document["labeled_spans"].predictions
    relations = document["binary_relations"].predictions
    entities_expected = document["labeled_spans"]
    relations_expected = document["binary_relations"]

    assert len(entities) == len(entities_expected) == 3
    assert len(relations) == len(relations_expected) == 1

    assert [
        entity.targets == expected_entity.targets
        for entity, expected_entity in zip(entities, entities_expected)
    ]
    assert [
        relation.targets == expected_relation.targets
        for relation, expected_relation in zip(relations, relations_expected)
    ]

    expected_scores = [
        0.9 * 0.95,
        0.85 * 0.9,
        0.8 * 0.85,
    ]

    assert allclose(
        tensor([entity.score for entity in entities]),
        tensor(expected_scores),
    )


def test_collate(taskmodule, task_encoding, save_batch_data=False):
    batch, targets = taskmodule.collate([task_encoding, task_encoding])
    assert targets is None

    batch_path = FIXTURES_ROOT / "taskmodules/batch.pt"
    batch_readable = FIXTURES_ROOT / "taskmodules/batch.json"

    if save_batch_data:
        save(batch, batch_path)
        serializable_batch = {k: v.cpu().numpy().tolist() for k, v in batch.items()}
        # Save readable json to inspect
        with open(batch_readable, "w+") as f:
            json.dump(serializable_batch, f, indent=2)

    batch_expected = load(batch_path)


def test_collate_with_targets(taskmodule, task_encoding, save_batch_data=False):
    encodings_with_targets = taskmodule.encode_targets(
        task_encodings=[task_encoding, task_encoding]
    )
    batch, targets = taskmodule.collate(encodings_with_targets)

    batch_path = FIXTURES_ROOT / "taskmodules/batch.pt"
    targets_path = FIXTURES_ROOT / "taskmodules/targets.pt"
    batch_readable = FIXTURES_ROOT / "taskmodules/batch.json"
    targets_readable = FIXTURES_ROOT / "taskmodules/targets.json"

    if save_batch_data:
        save(batch, batch_path)
        save(targets, targets_path)
        serializable_batch = {k: v.cpu().numpy().tolist() for k, v in batch.items()}
        # Save readable jsons to inspect
        with open(batch_readable, "w+") as f:
            json.dump(serializable_batch, f, indent=2)
        serializable_targets = {k: v.cpu().numpy().tolist() for k, v in targets.items()}
        with open(targets_readable, "w+") as f:
            json.dump(serializable_targets, f, indent=2)

    batch_expected = load(batch_path)
    targets_expected = load(targets_path)

    for k in batch.keys():
        assert allclose(batch[k], batch_expected[k])
    for k in targets.keys():
        assert allclose(targets[k], targets_expected[k])


@pytest.fixture(scope="module")
def model_output():
    return {
        "rel_head": tensor([[2, 1], [0, -1]]),
        "rel_tail": tensor([[1, 2], [1, -1]]),
        "rel_probability": tensor([[1.0, 0.5], [0.0, -1.0]]),
        "rel_label": tensor([[0, 1], [0, -1]]),
        "rel_label_probability": tensor([[0.5, 1.0], [0.5, -1.0]]),
        "span_start": tensor([[1, 26, 41], [1, 26, -1]]),
        "span_end": tensor([[26, 41, 76], [26, 41, -1]]),
        "span_probability": tensor([[0.5, 0.25, 1.0], [0.5, 0.25, -1.0]]),
        "span_label": tensor([[0, 2, 1], [0, 2, -1]]),
        "span_label_probability": tensor([[0.5, 1.0, 0.75], [0.5, 1.0, -1.0]]),
    }


def test_collate_and_unbatch_output(taskmodule, task_encoding):
    encodings_with_targets = taskmodule.encode_targets(
        task_encodings=[task_encoding, task_encoding]
    )
    _, targets = taskmodule.collate(encodings_with_targets)
    task_output = taskmodule.unbatch_output(model_output=targets)
    assert task_output == [
        TaskOutputType(
            span_start=[1, 26, 41],
            span_end=[26, 41, 76],
            span_label=[0, 2, 1],
            rel_head=[2],
            rel_tail=[1],
            rel_label=[0],
            span_probability=None,
            span_label_probability=None,
            rel_probability=None,
            rel_label_probability=None,
        ),
        TaskOutputType(
            span_start=[1, 26, 41],
            span_end=[26, 41, 76],
            span_label=[0, 2, 1],
            rel_head=[2],
            rel_tail=[1],
            rel_label=[0],
            span_probability=None,
            span_label_probability=None,
            rel_probability=None,
            rel_label_probability=None,
        ),
    ]


def test_unbatch(taskmodule, model_output):
    task_output = taskmodule.unbatch_output(model_output)
    assert task_output == [
        TaskOutputType(
            span_start=[1, 26, 41],
            span_end=[26, 41, 76],
            span_label=[0, 2, 1],
            rel_head=[2, 1],
            rel_tail=[1, 2],
            rel_label=[0, 1],
            span_probability=[0.5, 0.25, 1.0],
            span_label_probability=[0.5, 1.0, 0.75],
            rel_probability=[1.0, 0.5],
            rel_label_probability=[0.5, 1.0],
        ),
        TaskOutputType(
            span_start=[1, 26],
            span_end=[26, 41],
            span_label=[0, 2],
            rel_head=[0],
            rel_tail=[1],
            rel_label=[0],
            span_probability=[0.5, 0.25],
            span_label_probability=[0.5, 1.0],
            rel_probability=[0.0],
            rel_label_probability=[0.5],
        ),
    ]


def test_encode_decode(taskmodule, document):
    encodings = taskmodule.encode(document, encode_target=True)
    targets = [encoding.targets for encoding in encodings]
    decoded = taskmodule.decode(encodings, targets, inplace=True)

    assert len(decoded) == 1

    decoded = decoded[0]

    assert (
        decoded.text
        == "Recently, courts have held that debt collectors can escape 1692i's venue provisions entirely by pursuing debt collection through arbitration instead. As the NAF studies reflect, arbitration has not proven a satisfactory alternative. I urge the CFPB to include in a rule language interpreting 1692i as requiring debt collectors to proceed in court, not through largely-unregulated arbitral forums."
    )

    spans = list(map(lambda x: x.resolve(), decoded["labeled_spans"]))
    spans_pred = list(map(lambda x: x.resolve(), decoded["labeled_spans"].predictions))
    assert spans_pred == [
        (
            "fact",
            "Recently, courts have held that debt collectors can escape 1692i's venue provisions entirely by pursuing debt collection through arbitration instead.",
        ),
        (
            "value",
            "As the NAF studies reflect, arbitration has not proven a satisfactory alternative.",
        ),
        (
            "policy",
            "I urge the CFPB to include in a rule language interpreting 1692i as requiring debt collectors to proceed in court, not through largely-unregulated arbitral forums.",
        ),
    ]
    assert spans == spans_pred

    rels = list(map(lambda x: x.resolve(), decoded["binary_relations"]))
    rels_pred = list(map(lambda x: x.resolve(), decoded["binary_relations"].predictions))

    assert rels_pred == [
        (
            "reason",
            (
                (
                    "policy",
                    "I urge the CFPB to include in a rule language interpreting 1692i as requiring debt collectors to proceed in court, not through largely-unregulated arbitral forums.",
                ),
                (
                    "value",
                    "As the NAF studies reflect, arbitration has not proven a satisfactory alternative.",
                ),
            ),
        )
    ]
    assert rels_pred == rels


def test_long_document_encoding():
    taskmodule = AMTaskModule(
        tokenizer_name_or_path="allenai/longformer-base-4096",
        tokenize_kwargs={"max_length": 10, "stride": 1},
    )
    with open("tests/fixtures/datasets/json/train.json") as f:
        doc = [d for d in json.load(f)["data"] if d["id"] == "train_doc5"][0]

    document = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(doc["text"])
    labeled_spans = [LabeledSpan(**dct) for dct in doc["entities"]]
    binary_relations = [
        BinaryRelation(labeled_spans[rel["head"]], labeled_spans[rel["tail"]], rel["label"])
        for rel in doc["relations"]
    ]
    for span in labeled_spans:
        document.labeled_spans.append(span)
    for rel in binary_relations:
        document.binary_relations.append(rel)
    taskmodule.prepare([document])

    encoded = taskmodule.encode(document, encode_target=False)

    assert len(encoded) == 2

    for chunk in encoded:
        input_ids = chunk.inputs.ids
        assert len(input_ids) <= taskmodule.tokenizer.model_max_length


def test_encode_decode_long_document():
    taskmodule = AMTaskModule(
        tokenizer_name_or_path="allenai/longformer-base-4096",
        tokenize_kwargs={"max_length": 10, "stride": 1},
    )
    with open(FIXTURES_ROOT / "datasets/json/train.json") as f:
        doc = [d for d in json.load(f)["data"] if d["id"] == "train_doc5"][0]

    document = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(doc["text"])
    labeled_spans = [LabeledSpan(**dct) for dct in doc["entities"]]
    binary_relations = [
        BinaryRelation(labeled_spans[rel["head"]], labeled_spans[rel["tail"]], rel["label"])
        for rel in doc["relations"]
    ]
    for span in labeled_spans:
        document.labeled_spans.append(span)
    for rel in binary_relations:
        document.binary_relations.append(rel)

    taskmodule.prepare([document])
    encodings = taskmodule.encode(document, encode_target=True)
    targets = [encoding.targets for encoding in encodings]
    decoded = taskmodule.decode(encodings, targets, inplace=True)
    decoded_doc: TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions = decoded[0]
    # Annotations may be duplicated due to stride != 0
    decoded_doc = decoded_doc.deduplicate_annotations()
    assert len(decoded) == 1
    assert len(encodings) == 2

    assert decoded_doc.text == "First sentence. Entity G works at H. And founded I."

    spans = list(map(lambda x: x.resolve(), decoded_doc["labeled_spans"]))
    spans_pred = list(map(lambda x: x.resolve(), decoded_doc["labeled_spans"].predictions))
    assert spans_pred == [("PER", "Entity G"), ("ORG", "H"), ("ORG", "I")]
    assert spans == spans_pred

    rels = list(map(lambda x: x.resolve(), decoded_doc["binary_relations"]))
    rels_pred = list(map(lambda x: x.resolve(), decoded_doc["binary_relations"].predictions))
    assert rels == [
        ("per:employee_of", (("PER", "Entity G"), ("ORG", "H"))),
        ("per:founder", (("PER", "Entity G"), ("ORG", "I"))),
        ("org:founded_by", (("ORG", "I"), ("ORG", "H"))),
    ]
    assert rels_pred == [
        ("per:employee_of", (("PER", "Entity G"), ("ORG", "H"))),
        # Entities were too far away and did not fit into one window, so relation was not created
        # ('per:founder', (('PER', 'Entity G'), ('ORG', 'I'))),
        ("org:founded_by", (("ORG", "I"), ("ORG", "H"))),
    ]
    assert rels_pred != rels
