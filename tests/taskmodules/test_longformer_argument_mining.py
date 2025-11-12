import logging

import pytest
import torch
from pie_core import TaskModule
from pie_datasets import DatasetDict, load_dataset
from torch import tensor

from src.taskmodules import AMTaskModule

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def documents():
    dataset = load_dataset("pie/cdcp")
    documents = dataset["test"][:3]
    return documents


@pytest.fixture(scope="module")
def document(documents):
    return documents[0]


@pytest.fixture(scope="module")
def taskmodule():
    return AMTaskModule("allenai/longformer-base-4096")


# TODO: move these raw things to separate file
# batch model input


# Pipeline:
# batch -> Model.encode -> exs, mask #
#    -> Model.span_output -> span_out
#       -> Model.span_prdict -> span_pred
#            -> model.proposition_output -> prop_out
#            -> model.edge_output        -> arc_out, rel_out
# -> span_exs, span_mask, span, prop, arc, rel


def test_encode_input(taskmodule, document):
    logger.warning(document)
    encoded_input = taskmodule.encode_input(document)
    assert encoded_input[0].tokens == ""


def test_encode(taskmodule, documents):
    assert len(documents) == 3
    logger.warning(documents)
