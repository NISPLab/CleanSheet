import numpy as np
import pytest

from .utils import FakeDataset, build_fake_dataset

FLOWERS102_TESTSET_NUM = 5 * 102


def build_flowers102_fake_dataset(dataset_type: str, **kwargs) -> FakeDataset:
    dataset_cfg = dict(type=dataset_type.replace('Flowers102', 'FakeDataset'),
                       x_shape=(3, 32, 32),
                       y_range=(0, 101),
                       nums=FLOWERS102_TESTSET_NUM,
                       **kwargs)

    return build_fake_dataset(dataset_cfg)


@pytest.mark.parametrize('dataset_type', ['Flowers102'])
def test_xy(dataset_type: str) -> None:
    flowers102 = build_flowers102_fake_dataset(dataset_type)

    xy = flowers102.get_xy()
    x, y = xy
    assert len(x) == len(y)
    assert isinstance(y[0], int)

    old_x = x.copy()
    old_y = y.copy()

    flowers102.set_xy(xy)
    assert all(
        [np.array_equal(nx, ox) for nx, ox in zip(flowers102.data, old_x)])
    assert flowers102.targets == old_y

    x = x[:flowers102.num_classes]
    y = y[:flowers102.num_classes]
    flowers102.set_xy((x, y))
    assert all([np.array_equal(nx, ox) for nx, ox in zip(flowers102.data, x)])
    assert flowers102.targets == y
    assert flowers102.num_classes == len(set(y))
    assert len(flowers102.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['PoisonLabelFlowers102'])
@pytest.mark.parametrize('poison_label', [-1, 5, 102])
def test_poison_label(poison_label: int, dataset_type: str) -> None:
    kwargs = dict(poison_label=poison_label)

    if poison_label < 0 or poison_label >= 43:
        with pytest.raises(ValueError):
            _ = build_flowers102_fake_dataset(dataset_type, **kwargs)
        return
    flowers102 = build_flowers102_fake_dataset(dataset_type, **kwargs)
    assert flowers102.poison_label == poison_label

    assert flowers102.num_classes == 1
    assert all(map(lambda x: x == poison_label, flowers102.targets))
    assert len(flowers102.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['RatioPoisonLabelFlowers102'])
@pytest.mark.parametrize('poison_label', [-1, 5, 102])
@pytest.mark.parametrize('ratio', [0, 0.2, 1, 1.2])
def test_ratio_poison_label(ratio: float, poison_label: int,
                            dataset_type: str) -> None:
    kwargs = dict(ratio=ratio, poison_label=poison_label)

    if (poison_label < 0 or poison_label >= 102) or \
            (ratio <= 0 or ratio > 1):
        with pytest.raises(ValueError):
            _ = build_flowers102_fake_dataset(dataset_type, **kwargs)
        return
    flowers102 = build_flowers102_fake_dataset(dataset_type, **kwargs)
    assert flowers102.poison_label == poison_label

    assert len(flowers102) == round(FLOWERS102_TESTSET_NUM * ratio)
    assert flowers102.num_classes == 1
    assert all(map(lambda x: x == poison_label, flowers102.targets))
    assert len(flowers102.data.shape) == 4
