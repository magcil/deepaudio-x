from pathlib import Path

import numpy as np
import pytest
import torch

import deepaudiox as dax
from deepaudiox.modules.backbones import BACKBONES
from deepaudiox.modules.classifier.classifier import MLPHead
from deepaudiox.utils.training_utils import (
    get_class_mapping_from_dir,
    get_class_mapping_from_list,
    random_split_audio_dataset,
)

TRAIN_DIR = Path(__file__).parent / "testing_dataset" / "train"
VALIDATION_DIR = Path(__file__).parent / "testing_dataset" / "validation"
TEST_DIR = Path(__file__).parent / "testing_dataset" / "test"


def get_test_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


TEST_DEVICE = get_test_device()


@pytest.fixture(scope="session")
def file_to_class_mapping():
    return {
        Path(__file__).parent / "testing_dataset" / "train" / "gaussiannoise" / "gaussiannoise_0.wav": "gaussiannoise",
        Path(__file__).parent / "testing_dataset" / "train" / "sawwave" / "sawwave_0.wav": "sawwave",
        Path(__file__).parent / "testing_dataset" / "train" / "sinewave" / "sinewave_0.wav": "sinewave",
        Path(__file__).parent / "testing_dataset" / "train" / "squarewave" / "squarewave_0.wav": "squarewave",
        Path(__file__).parent / "testing_dataset" / "train" / "trianglewave" / "trianglewave_0.wav": "trianglewave",
    }


def test_class_mapping():
    class_mapping_dir = get_class_mapping_from_dir(str(TRAIN_DIR))
    class_mapping_list = get_class_mapping_from_list(
        ["gaussiannoise", "sawwave", "sinewave", "squarewave", "trianglewave"]
    )

    expected = {
        "gaussiannoise": 0,
        "sawwave": 1,
        "sinewave": 2,
        "squarewave": 3,
        "trianglewave": 4,
    }

    assert isinstance(class_mapping_dir, dict)
    assert isinstance(class_mapping_list, dict)

    assert class_mapping_dir == expected
    assert class_mapping_list == expected


@pytest.mark.parametrize("sample_rate", [16000, 22000])
def test_dataset_loading_from_dir(file_to_class_mapping, sample_rate):
    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))
    dataset = dax.audio_classification_dataset_from_dir(
        root_dir=str(TRAIN_DIR), sample_rate=sample_rate, class_mapping=class_mapping
    )

    assert len(dataset) == 100

    item = dataset[0]
    assert isinstance(item["feature"], np.ndarray)
    assert len(item["feature"]) == int(5 * sample_rate)
    assert isinstance(item["y_true"], int)
    assert isinstance(item["class_name"], str)


@pytest.mark.parametrize("sample_rate", [16000, 22000])
def test_dataset_loading_from_dict(file_to_class_mapping, sample_rate):
    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))
    dataset = dax.audio_classification_dataset_from_dictionary(
        file_to_class_mapping=file_to_class_mapping, sample_rate=sample_rate, class_mapping=class_mapping
    )

    assert len(dataset) == 5

    item = dataset[0]
    assert isinstance(item["feature"], np.ndarray)
    assert len(item["feature"]) == int(5 * sample_rate)
    assert isinstance(item["y_true"], int)
    assert isinstance(item["class_name"], str)


def test_dataset_splitting():
    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))
    dataset = dax.audio_classification_dataset_from_dir(
        root_dir=str(TRAIN_DIR), sample_rate=16000, class_mapping=class_mapping
    )

    train_dataset, validation_dataset = random_split_audio_dataset(dataset=dataset, train_ratio=0.8)

    assert len(train_dataset) == 80
    assert len(validation_dataset) == 20


@pytest.mark.parametrize("segment_duration", [0.5, 1.0])
def test_dataset_segmentation(segment_duration):
    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))
    dataset = dax.audio_classification_dataset_from_dir(
        root_dir=str(TRAIN_DIR), sample_rate=16000, class_mapping=class_mapping, segment_duration=segment_duration
    )

    expected_num_items = 100 * (5 // segment_duration)
    assert len(dataset) == expected_num_items


@pytest.mark.parametrize("input_tensor", [torch.randn(1, 16000)])
def test_beats_backbone(input_tensor):
    backbone = BACKBONES["beats"]()

    output = backbone.forward_pipeline(input_tensor)
    assert output.shape == (1, 48, 768)


@pytest.mark.parametrize("input_tensor", [torch.randn(1, 32000)])
def test_passt_backbone(input_tensor):
    passt = BACKBONES["passt"]()

    output = passt.forward_pipeline(input_tensor)
    print(output.shape)

    assert output.shape == (1, 108, 768)


@pytest.mark.parametrize("input_tensor", [torch.randn(1, 768)])
@pytest.mark.parametrize("hidden_layers", [None, [128], [512, 128]])
def test_mlp_head(input_tensor, hidden_layers):
    head = MLPHead(num_classes=4, in_dim=768, hidden_layers=hidden_layers, activation="relu", apply_batch_norm=False)

    output = head(input_tensor)
    assert output.shape == (1, 4)


@pytest.mark.parametrize("x", [torch.randn(1, 5 * 16_000)])
@pytest.mark.parametrize("backbone", ["beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as"])
@pytest.mark.parametrize("pooling", ["simpool", "gap", "ep"])
@pytest.mark.parametrize("freeze_backbone", [True, False])
@pytest.mark.parametrize("pretrained", [True, False])
def test_backbone_constructor(backbone, pretrained, freeze_backbone, pooling, x):
    backbone_constructor = dax.Backbone(
        backbone=backbone, pretrained=pretrained, freeze_backbone=freeze_backbone, pooling=pooling, sample_rate=16_000
    )

    # Test feature forward
    with torch.no_grad():
        z = backbone_constructor(x)
        if z.dim() == 3:
            assert z.shape[-1] == backbone_constructor.out_dim
        elif z.dim() == 4:
            assert z.shape[1] == backbone_constructor.out_dim
        else:
            raise AssertionError(f"Unexpected backbone output rank: {z.dim()}")
        w = backbone_constructor.forward_with_pooling(x)
        assert w.shape == (1, backbone_constructor.out_dim)

        if freeze_backbone:
            assert not any(p.requires_grad for p in backbone_constructor.backbone.parameters())


@pytest.mark.parametrize("input_tensor", [torch.randn(1, 16000)])
@pytest.mark.parametrize("num_classes", [2, 8])
@pytest.mark.parametrize("backbone", ["beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as"])
@pytest.mark.parametrize("pooling", ["simpool", "gap", "ep"])
@pytest.mark.parametrize("freeze_backbone", [True, False])
@pytest.mark.parametrize("pretrained", [True, False])
def test_model_construction(input_tensor, num_classes, backbone, pooling, freeze_backbone, pretrained):
    model = dax.AudioClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pooling=pooling,
        freeze_backbone=freeze_backbone,
        sample_rate=16000,
        classifier_hidden_layers=None,
        activation="relu",
        apply_batch_norm=False,
        pretrained=pretrained,
    )

    # Test output
    output = model(input_tensor)
    assert output.shape == (1, num_classes)

    # Test embedding extraction
    embeddings = model.forward_with_pooling(input_tensor)
    assert embeddings.shape == (1, model.backbone_constructor.out_dim)

    if freeze_backbone:
        assert not any(p.requires_grad for p in model.backbone_constructor.backbone.parameters())


def test_training_loop():
    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))

    train_dataset = dax.audio_classification_dataset_from_dir(
        root_dir=str(TRAIN_DIR), sample_rate=16000, class_mapping=class_mapping
    )
    validation_dataset = dax.audio_classification_dataset_from_dir(
        root_dir=str(VALIDATION_DIR), sample_rate=16000, class_mapping=class_mapping
    )

    model = dax.AudioClassifier(
        num_classes=5,
        backbone="beats",
        pooling="gap",
        freeze_backbone=True,
        sample_rate=16000,
        classifier_hidden_layers=None,
        activation="relu",
        apply_batch_norm=False,
        pretrained=True,
    )

    trainer = dax.Trainer(
        train_dset=train_dataset,
        model=model,
        validation_dset=validation_dataset,
        learning_rate=1e-3,
        epochs=3,
        patience=5,
        num_workers=4,
        batch_size=16,
        path_to_checkpoint="testing_checkpoint.pt",
        device=TEST_DEVICE,
    )

    trainer.train()

    train_loss = np.array(trainer.state.train_loss)
    val_loss = np.array(trainer.state.validation_loss)

    assert len(train_loss) == 3
    assert len(val_loss) == 3
    assert train_loss[2] <= train_loss[1]
    assert val_loss[2] < val_loss[1]
    assert (Path(__file__).resolve().parents[1] / "testing_checkpoint.pt").exists()


def test_evaluation_loop():
    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))

    test_dataset = dax.audio_classification_dataset_from_dir(
        root_dir=str(TEST_DIR), sample_rate=16000, class_mapping=class_mapping
    )

    model = dax.AudioClassifier(
        num_classes=5,
        backbone="beats",
        pooling="gap",
        freeze_backbone=True,
        sample_rate=16000,
        classifier_hidden_layers=None,
        activation="relu",
        apply_batch_norm=False,
        pretrained=True,
    )

    path_to_checkpoint = Path(__file__).resolve().parents[1] / "testing_checkpoint.pt"
    model = dax.AudioClassifier.from_checkpoint(str(path_to_checkpoint))

    evaluator = dax.Evaluator(
        test_dset=test_dataset, model=model, num_workers=4, batch_size=16, class_mapping=class_mapping, device=TEST_DEVICE
    )

    evaluator.evaluate()

    y_true = evaluator.state.y_true
    y_pred = evaluator.state.y_pred
    posteriors = evaluator.state.posteriors

    assert len(y_true) == 25
    assert len(y_pred) == 25
    assert len(posteriors) == 25


@pytest.mark.parametrize(
    "path_to_test_file",
    ["tests/testing_dataset/train/sinewave/sinewave_0.wav", "tests/testing_dataset/train/sawwave/sawwave_0.wav"],
)
def test_inference(path_to_test_file: str):
    path_to_checkpoint = Path(__file__).resolve().parents[1] / "testing_checkpoint.pt"
    path_wav = Path(__file__).resolve().parents[1] / path_to_test_file

    model = dax.AudioClassifier.from_checkpoint(str(path_to_checkpoint))

    class_mapping = get_class_mapping_from_dir(str(TRAIN_DIR))

    # Inference on the whole audio file
    inference = model.inference_on_file(path=path_wav, class_mapping=class_mapping, sample_rate=16_000)

    final_label = path_wav.parent.name

    assert inference["final_label"] == final_label

    # Inference with segmentation
    inference_segmented = model.inference_on_file(
        path=path_wav, class_mapping=class_mapping, sample_rate=16_000, segment_duration=1.0
    )

    assert len(inference_segmented["segment_labels"]) == 5

    # Inference with segmentation
    inference_segmented = model.inference_on_file(
        path=path_wav, class_mapping=class_mapping, sample_rate=16_000, segment_duration=2.0
    )
