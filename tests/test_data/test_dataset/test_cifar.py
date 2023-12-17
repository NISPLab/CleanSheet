import numpy as np
import pytest

from .utils import FakeDataset, build_fake_dataset

CIFAR_TESTSET_NUM = 1000


def build_cifar_fake_dataset(dataset_type: str, **kwargs) -> FakeDataset:
    if dataset_type.endswith('CIFAR100'):
        y_range = (0, 99)
        dataset_type = dataset_type.replace('CIFAR100', 'FakeDataset')
    else:
        y_range = (0, 9)
        dataset_type = dataset_type.replace('CIFAR10', 'FakeDataset')

    dataset_cfg = dict(type=dataset_type,
                       x_shape=(3, 32, 32),
                       y_range=y_range,
                       nums=CIFAR_TESTSET_NUM,
                       **kwargs)

    return build_fake_dataset(dataset_cfg)


@pytest.mark.parametrize('dataset_type', ['CIFAR10', 'CIFAR100'])
def test_xy(dataset_type: str) -> None:
    cifar = build_cifar_fake_dataset(dataset_type)

    xy = cifar.get_xy()
    x, y = xy
    assert len(x) == len(y)
    assert isinstance(y[0], int)

    old_x = x.copy()
    old_y = y.copy()

    cifar.set_xy(xy)
    assert all([np.array_equal(nx, ox) for nx, ox in zip(cifar.data, old_x)])
    assert cifar.targets == old_y

    x = x[:cifar.num_classes]
    y = y[:cifar.num_classes]
    cifar.set_xy((x, y))
    assert all([np.array_equal(nx, ox) for nx, ox in zip(cifar.data, x)])
    assert cifar.targets == y
    assert cifar.num_classes == len(set(y))
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize(['start_idx', 'end_idx', 'dataset_type'],
                         [(0, 9, 'IndexCIFAR10'), (-10, 8, 'IndexCIFAR10'),
                          (2, 12, 'IndexCIFAR10'), (4, 4, 'IndexCIFAR10'),
                          (4, 3, 'IndexCIFAR10'), (0, 99, 'IndexCIFAR100'),
                          (-10, 8, 'IndexCIFAR100'),
                          (40, 50, 'IndexCIFAR100')])
def test_index(start_idx: int, end_idx: int, dataset_type: str) -> None:
    kwargs = dict(start_idx=start_idx, end_idx=end_idx)

    if start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return
    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)
    assert cifar.start_idx == start_idx
    assert cifar.end_idx == end_idx

    for y in cifar.targets:
        assert start_idx <= y <= end_idx

    assert cifar.num_classes == min(
        cifar.end_idx, cifar.raw_num_classes - 1) - max(cifar.start_idx, 0) + 1
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize(['ratio', 'dataset_type'], [(-1, 'RatioCIFAR10'),
                                                     (0, 'RatioCIFAR10'),
                                                     (0.1, 'RatioCIFAR10'),
                                                     (0.5, 'RatioCIFAR10'),
                                                     (1, 'RatioCIFAR10'),
                                                     (2, 'RatioCIFAR10'),
                                                     (0.4, 'RatioCIFAR100')])
def test_ratio(ratio: float, dataset_type: str) -> None:
    kwargs = dict(ratio=ratio)

    if ratio <= 0 or ratio > 1:
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return
    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)

    assert cifar.num_classes == cifar.raw_num_classes
    assert len(cifar.targets) == \
        int(CIFAR_TESTSET_NUM / cifar.num_classes * ratio) * cifar.num_classes
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize('range_ratio', [(-1, 0.2), (0, 2), (0.1, 0.1),
                                         (0.5, 0.2), (0.1, 0.5), (0, 1)])
@pytest.mark.parametrize('dataset_type',
                         ['RangeRatioCIFAR10', 'RangeRatioCIFAR100'])
def test_range_ratio(range_ratio: tuple[float, float],
                     dataset_type: str) -> None:
    kwargs = dict(range_ratio=range_ratio)

    start_ratio = range_ratio[0]
    end_ratio = range_ratio[1]
    if not (0 <= start_ratio < end_ratio <= 1):
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return

    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)
    assert cifar.num_classes == cifar.raw_num_classes
    assert len(cifar.targets) == \
        round(CIFAR_TESTSET_NUM * (end_ratio - start_ratio))
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize(['range_ratio1', 'range_ratio2'],
                         [((0, 0.5), (0.5, 1)), ((0, 0.6), (0.4, 1)),
                          ((0, 0.7), (0.3, 1)), ((0, 0.5), (0, 1))])
@pytest.mark.parametrize('dataset_type',
                         ['RangeRatioCIFAR10', 'RangeRatioCIFAR100'])
def test_range_ratio_intersection(range_ratio1: tuple[float, float],
                                  range_ratio2: tuple[float, float],
                                  dataset_type: str) -> None:

    cifar1 = build_cifar_fake_dataset(dataset_type=dataset_type,
                                      range_ratio=range_ratio1,
                                      cache_xy=True)
    cifar2 = build_cifar_fake_dataset(dataset_type=dataset_type,
                                      range_ratio=range_ratio2,
                                      cache_xy=True)

    cat_x = np.concatenate([cifar1.data, cifar2.data], axis=0)
    unique_x = np.unique(cat_x, axis=0)
    intersection_number = cat_x.shape[0] - unique_x.shape[0]

    intersection_ratio = max(0, range_ratio1[1] - range_ratio2[0])
    assert round(intersection_ratio * CIFAR_TESTSET_NUM) == intersection_number


@pytest.mark.parametrize('dataset_type',
                         ['IndexRatioCIFAR10', 'IndexRatioCIFAR100'])
@pytest.mark.parametrize(['start_idx', 'end_idx', 'ratio'], [(4, 3, 0.5),
                                                             (3, 4, 0),
                                                             (3, 4, 2),
                                                             (1, 4, 0.1)])
def test_index_ratio(start_idx: int, end_idx: int, ratio: float,
                     dataset_type: str) -> None:
    kwargs = dict(start_idx=start_idx, end_idx=end_idx, ratio=ratio)

    if ratio <= 0 or ratio > 1 or start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return
    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)
    assert cifar.start_idx == start_idx
    assert cifar.end_idx == end_idx

    for y in cifar.targets:
        assert start_idx <= y <= end_idx
    assert len(cifar.targets) == \
        cifar.num_classes / cifar.raw_num_classes * CIFAR_TESTSET_NUM * ratio
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize('dataset_type',
                         ['PoisonLabelCIFAR10', 'PoisonLabelCIFAR100'])
@pytest.mark.parametrize('poison_label', [-1, 5, 101])
def test_poison_label(poison_label: int, dataset_type: str) -> None:
    kwargs = dict(poison_label=poison_label)

    if poison_label < 0 or poison_label >= 100:
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return
    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)
    assert cifar.poison_label == poison_label

    assert cifar.num_classes == 1
    assert all(map(lambda x: x == poison_label, cifar.targets))
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize(
    'dataset_type', ['RatioPoisonLabelCIFAR10', 'RatioPoisonLabelCIFAR100'])
@pytest.mark.parametrize('poison_label', [-1, 5, 101])
@pytest.mark.parametrize('ratio', [0, 0.2, 1, 1.2])
def test_ratio_poison_label(ratio: float, poison_label: int,
                            dataset_type: str) -> None:
    kwargs = dict(ratio=ratio, poison_label=poison_label)

    if (poison_label < 0 or poison_label >= 100) or \
            (ratio <= 0 or ratio > 1):
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return
    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)
    assert cifar.poison_label == poison_label

    assert len(cifar) == round(CIFAR_TESTSET_NUM * ratio)
    assert cifar.num_classes == 1
    assert all(map(lambda x: x == poison_label, cifar.targets))
    assert len(cifar.data.shape) == 4


@pytest.mark.parametrize(
    'dataset_type',
    ['RangeRatioPoisonLabelCIFAR10', 'RangeRatioPoisonLabelCIFAR100'])
@pytest.mark.parametrize('poison_label', [-1, 1, 101])
@pytest.mark.parametrize('range_ratio', [(0, 0.2), (0.2, 0.5), (0.5, 1),
                                         (0.5, 0.2)])
def test_range_ratio_poison_label(range_ratio: tuple[float,
                                                     float], poison_label: int,
                                  dataset_type: str) -> None:
    kwargs = dict(range_ratio=range_ratio, poison_label=poison_label)

    if poison_label < 0 or poison_label >= 100 or \
            not (0 <= range_ratio[0] < range_ratio[1] <= 1):
        with pytest.raises(ValueError):
            _ = build_cifar_fake_dataset(dataset_type, **kwargs)
        return
    cifar = build_cifar_fake_dataset(dataset_type, **kwargs)
    assert cifar.poison_label == poison_label

    assert len(cifar) == round(CIFAR_TESTSET_NUM *
                               (range_ratio[1] - range_ratio[0]))
    assert cifar.num_classes == 1
    assert all(map(lambda x: x == poison_label, cifar.targets))
    assert len(cifar.data.shape) == 4
