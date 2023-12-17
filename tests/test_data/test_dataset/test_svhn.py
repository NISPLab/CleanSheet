import numpy as np
import pytest

from .utils import FakeDataset, build_fake_dataset

SVHN_TESTSET_NUM = 10 * 100


def build_svhn_fake_dataset(dataset_type: str, **kwargs) -> FakeDataset:
    dataset_cfg = dict(type=dataset_type.replace('SVHN', 'FakeDataset'),
                       x_shape=(3, 32, 32),
                       y_range=(0, 9),
                       nums=SVHN_TESTSET_NUM,
                       **kwargs)

    return build_fake_dataset(dataset_cfg)


@pytest.mark.parametrize('dataset_type', ['SVHN'])
def test_xy(dataset_type: str) -> None:
    svhn = build_svhn_fake_dataset(dataset_type)

    xy = svhn.get_xy()
    x, y = xy
    assert len(x) == len(y)
    assert isinstance(y[0], int)

    old_x = x.copy()
    old_y = y.copy()

    svhn.set_xy(xy)
    assert all([np.array_equal(nx, ox) for nx, ox in zip(svhn.data, old_x)])
    assert svhn.targets == old_y

    x = x[:svhn.num_classes]
    y = y[:svhn.num_classes]
    svhn.set_xy((x, y))
    assert all([np.array_equal(nx, ox) for nx, ox in zip(svhn.data, x)])
    assert svhn.targets == y
    assert svhn.num_classes == len(set(y))
    assert len(svhn.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['PoisonLabelSVHN'])
@pytest.mark.parametrize('poison_label', [-1, 5, 43])
def test_poison_label(poison_label: int, dataset_type: str) -> None:
    kwargs = dict(poison_label=poison_label)

    if poison_label < 0 or poison_label >= 43:
        with pytest.raises(ValueError):
            _ = build_svhn_fake_dataset(dataset_type, **kwargs)
        return
    svhn = build_svhn_fake_dataset(dataset_type, **kwargs)
    assert svhn.poison_label == poison_label

    assert svhn.num_classes == 1
    assert all(map(lambda x: x == poison_label, svhn.targets))
    assert len(svhn.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['RatioPoisonLabelSVHN'])
@pytest.mark.parametrize('poison_label', [-1, 5, 10])
@pytest.mark.parametrize('ratio', [0, 0.2, 1, 1.2])
def test_ratio_poison_label(ratio: float, poison_label: int,
                            dataset_type: str) -> None:
    kwargs = dict(ratio=ratio, poison_label=poison_label)

    if (poison_label < 0 or poison_label >= 10) or \
            (ratio <= 0 or ratio > 1):
        with pytest.raises(ValueError):
            _ = build_svhn_fake_dataset(dataset_type, **kwargs)
        return
    svhn = build_svhn_fake_dataset(dataset_type, **kwargs)
    assert svhn.poison_label == poison_label

    assert len(svhn) == round(SVHN_TESTSET_NUM * ratio)
    assert svhn.num_classes == 1
    assert all(map(lambda x: x == poison_label, svhn.targets))
    assert len(svhn.data.shape) == 4
