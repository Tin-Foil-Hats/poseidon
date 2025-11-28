import math

import torch

from poseidon.training.lr_schedulers import LinearWarmupCosineAnnealingLR


def _collect_lrs(scheduler, steps: int) -> list[float]:
    values = []
    for _ in range(steps):
        scheduler.step()
        values.append(scheduler.get_last_lr()[0])
    return values


def test_linear_warmup_cosine_replicates_expected_curve():
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([param], lr=1e-3)
    sched = LinearWarmupCosineAnnealingLR(
        opt,
        warmup_epochs=2,
        max_epochs=5,
        warmup_start_lr=1e-5,
        eta_min=1e-6,
    )

    lrs = _collect_lrs(sched, steps=5)

                                                                     
    assert math.isclose(lrs[0], 1e-5, rel_tol=0.0, abs_tol=1e-8)
    assert lrs[1] > lrs[0]
    assert math.isclose(lrs[1], 1e-3, rel_tol=1e-6)

                                                               
    assert lrs[2] < lrs[1]
    assert lrs[3] < lrs[2]
    assert math.isclose(lrs[4], 1e-6, rel_tol=0.0, abs_tol=1e-9)
