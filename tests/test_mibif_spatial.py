"""Tests TDD WBS 7.4.3 pour MIBIF et les références spatiales."""

import numpy as np
import pytest

from tpv.features import MIBIFSelector
from tpv.preprocessing import SpatialReference


def test_mibif_selects_a_bounded_number_of_informative_features() -> None:
    rng = np.random.default_rng(101)
    labels = np.repeat([0, 1], 40)
    features = rng.standard_normal((80, 24))
    features[:, 7] = labels + rng.normal(0.0, 0.01, labels.size)
    selector = MIBIFSelector(k=4, random_state=42)

    selected = selector.fit_transform(features, labels)

    assert selected.shape == (80, 4)
    assert 7 in selector.selected_indices_
    assert np.array_equal(selected, features[:, selector.selected_indices_])


def test_mibif_rejects_feature_count_outside_defense_range() -> None:
    with pytest.raises(ValueError, match="between 4 and 16"):
        MIBIFSelector(k=3).fit(np.ones((10, 6)), np.tile([0, 1], 5))


def test_car_has_zero_instantaneous_channel_mean_without_mutating_input() -> None:
    rng = np.random.default_rng(102)
    epochs = rng.standard_normal((5, 4, 64))
    unchanged = epochs.copy()

    referenced = SpatialReference(method="car").fit_transform(epochs)

    assert np.allclose(referenced.mean(axis=1), 0.0)
    assert np.array_equal(epochs, unchanged)


def test_laplacian_uses_only_declared_neighbours() -> None:
    epochs = np.zeros((1, 3, 8))
    epochs[:, 0, :] = 3.0
    transformer = SpatialReference(
        method="laplacian",
        channel_names=("C3", "Cz", "C4"),
        neighbours={"C3": ("Cz",), "Cz": ("C3", "C4"), "C4": ("Cz",)},
    )

    referenced = transformer.fit_transform(epochs)

    assert np.allclose(referenced[0, 0], 3.0)
    assert np.allclose(referenced[0, 1], -1.5)
    assert np.allclose(referenced[0, 2], 0.0)


@pytest.mark.parametrize(
    ("selector", "features", "labels", "message"),
    [
        (MIBIFSelector(), np.ones((4, 5)), None, "y is required"),
        (MIBIFSelector(), np.ones((4, 5)), np.ones(3), "one label"),
        (MIBIFSelector(k=6), np.ones((4, 5)), np.ones(4), "cannot exceed"),
        (MIBIFSelector(), np.ones((4, 2, 2)), np.ones(4), "2D"),
        (
            MIBIFSelector(),
            np.array([[np.nan] * 12] * 4),
            np.ones(4),
            "finite",
        ),
    ],
)
def test_mibif_rejects_invalid_training_contracts(
    selector, features, labels, message
) -> None:
    with pytest.raises(ValueError, match=message):
        selector.fit(features, labels)


def test_mibif_transform_requires_fit_and_same_feature_count() -> None:
    selector = MIBIFSelector(k=4)
    with pytest.raises(Exception, match="not fitted"):
        selector.transform(np.ones((2, 5)))
    selector.fit(np.arange(60, dtype=float).reshape(10, 6), np.tile([0, 1], 5))
    with pytest.raises(ValueError, match="different number"):
        selector.transform(np.ones((2, 5)))


@pytest.mark.parametrize(
    ("transformer", "epochs", "message"),
    [
        (SpatialReference(method="invalid"), np.ones((2, 3, 4)), "method"),
        (SpatialReference(method="laplacian"), np.ones((2, 3, 4)), "requires"),
        (
            SpatialReference(
                method="laplacian",
                channel_names=("C3",),
                neighbours={"C3": ("Cz",)},
            ),
            np.ones((2, 3, 4)),
            "channel_names",
        ),
        (
            SpatialReference(
                method="laplacian",
                channel_names=("C3", "Cz", "C4"),
                neighbours={"C3": (), "Cz": ("C3",), "C4": ("Cz",)},
            ),
            np.ones((2, 3, 4)),
            "no laplacian neighbour",
        ),
        (SpatialReference(), np.ones((2, 3)), "3D"),
        (SpatialReference(), np.full((2, 3, 4), np.nan), "finite"),
    ],
)
def test_spatial_reference_rejects_invalid_contracts(
    transformer, epochs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        transformer.fit(epochs)


def test_spatial_reference_transform_guards_state_and_channels() -> None:
    epochs = np.ones((2, 3, 4))
    transformer = SpatialReference(method="none")
    with pytest.raises(ValueError, match="fitted"):
        transformer.transform(epochs)
    transformer.fit(epochs)
    copied = transformer.transform(epochs)
    assert np.array_equal(copied, epochs)
    assert copied is not epochs
    with pytest.raises(ValueError, match="different number"):
        transformer.transform(np.ones((2, 4, 4)))
