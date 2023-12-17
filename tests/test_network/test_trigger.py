import pytest
import torch

from anti_kd_backdoor.network.trigger import Trigger


@torch.no_grad()
@pytest.mark.parametrize('size', [32, 224])
def test_trigger_init(size: int) -> None:
    trigger = Trigger(size)
    assert trigger.size == size
    assert list(trigger.mask.shape) == [size, size]
    assert list(trigger.trigger.shape) == [3, size, size]


@torch.no_grad()
@pytest.mark.parametrize('size', [32, 224])
def test_trigger_forward(size: int) -> None:
    trigger = Trigger(size)

    x = torch.rand(10, 3, size, size)
    xp = trigger(x)
    assert xp.shape == x.shape

    # test effect of mask
    trigger.mask.fill_(0)
    xp = trigger(x)
    assert torch.equal(xp, x)

    trigger.mask.fill_(1)
    xp = trigger(x)
    for i in range(xp.size(0)):
        assert torch.equal(xp[i], trigger.trigger)
