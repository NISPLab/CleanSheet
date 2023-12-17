import numpy as np
import pytest

from .utils import FakeDataset, build_fake_dataset

GTSRB_TESTSET_NUM = 43 * 50


def build_gtsrb_fake_dataset(dataset_type: str, **kwargs) -> FakeDataset:
    dataset_cfg = dict(type=dataset_type.replace('GTSRB', 'FakeDataset'),
                       x_shape=(3, 32, 32),
                       y_range=(0, 42),
                       nums=GTSRB_TESTSET_NUM,
                       **kwargs)

    return build_fake_dataset(dataset_cfg)


@pytest.mark.parametrize('dataset_type', ['GTSRB'])
def test_xy(dataset_type: str) -> None:
    gtsrb = build_gtsrb_fake_dataset(dataset_type)

    xy = gtsrb.get_xy()
    x, y = xy
    assert len(x) == len(y)
    assert isinstance(y[0], int)

    old_x = x.copy()
    old_y = y.copy()

    gtsrb.set_xy(xy)
    assert all([np.array_equal(nx, ox) for nx, ox in zip(gtsrb.data, old_x)])
    assert gtsrb.targets == old_y

    x = x[:gtsrb.num_classes]
    y = y[:gtsrb.num_classes]
    gtsrb.set_xy((x, y))
    assert all([np.array_equal(nx, ox) for nx, ox in zip(gtsrb.data, x)])
    assert gtsrb.targets == y
    assert gtsrb.num_classes == len(set(y))
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['PoisonLabelGTSRB'])
@pytest.mark.parametrize('poison_label', [-1, 5, 43])
def test_poison_label(poison_label: int, dataset_type: str) -> None:
    kwargs = dict(poison_label=poison_label)

    if poison_label < 0 or poison_label >= 43:
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.poison_label == poison_label

    assert gtsrb.num_classes == 1
    assert all(map(lambda x: x == poison_label, gtsrb.targets))
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['RatioPoisonLabelGTSRB'])
@pytest.mark.parametrize('poison_label', [-1, 5, 43])
@pytest.mark.parametrize('ratio', [0, 0.2, 1, 1.2])
def test_ratio_poison_label(ratio: float, poison_label: int,
                            dataset_type: str) -> None:
    kwargs = dict(ratio=ratio, poison_label=poison_label)

    if (poison_label < 0 or poison_label >= 43) or \
            (ratio <= 0 or ratio > 1):
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.poison_label == poison_label

    assert len(gtsrb) == round(GTSRB_TESTSET_NUM * ratio)
    assert gtsrb.num_classes == 1
    assert all(map(lambda x: x == poison_label, gtsrb.targets))
    assert len(gtsrb.data.shape) == 4
