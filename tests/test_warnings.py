"""Tests for the pre-run checks and the `on_warning` policy."""
import warnings

import numpy as np
import pytest

from fp1d import solve
from fp1d.api import RunAborted
from fp1d.diagnostics import preflight_warnings
from fp1d.grid import make_grid

# lambda + 2*mu = 4.995 here, so Forward Euler cannot survive it.
UNSTABLE = dict(method='forward euler', drift='-x', diffusion=0.5,
                left=-5, right=5, dx=0.05, total_time=0.5, dt=0.01,
                bc='neumann')


def test_unstable_forward_euler_warns_by_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run = solve(**UNSTABLE)

    assert len(caught) == 1
    assert 'unstable' in str(caught[0].message)
    assert '4.995' in str(caught[0].message)
    # The run still happened - 'warn' means warn, not refuse.
    assert run.result.snapshots.shape[0] > 0


def test_warning_text_is_recorded_in_run_messages():
    """Whatever the policy, the reason is retrievable afterwards."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run = solve(**UNSTABLE)
    assert any('unstable' in m for m in run.messages)


def test_on_warning_raise_aborts_before_integrating():
    with pytest.raises(RunAborted, match='unstable'):
        solve(**UNSTABLE, on_warning='raise')


def test_on_warning_ignore_is_silent_but_still_logs():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run = solve(**UNSTABLE, on_warning='ignore')

    assert caught == []
    assert any('unstable' in m for m in run.messages)


def test_ask_does_not_block_without_a_terminal():
    """The mode that prompts must degrade when there is nobody to
    prompt. Under pytest stdin is captured, which is exactly the
    situation that would otherwise hang the suite forever.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run = solve(**UNSTABLE, on_warning='ask')

    assert len(caught) == 1
    assert run.result.snapshots.shape[0] > 0


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="on_warning must be one of"):
        solve(**UNSTABLE, on_warning='maybe')


def test_stable_run_produces_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run = solve(**{**UNSTABLE, 'dt': 0.0005})

    assert caught == []
    assert run.messages == []


@pytest.mark.parametrize('method', ['backward euler', 'euler-maruyama'])
def test_only_forward_euler_gets_the_stability_warning(method):
    """Backward Euler is unconditionally stable and Euler-Maruyama has no
    CFL condition, so the same dt must not warn for either.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solve(**{**UNSTABLE, 'method': method, 'trials': 200})

    assert caught == []

def test_periodic_mismatch_raises_rather_than_asking():
    """There is no meaningful 'proceed anyway' for a periodic seam whose
    coefficients disagree, so this is an error in every mode.
    """
    problem = dict(method='backward euler', drift='x', diffusion=0.5,
                   left=-2, right=3, dx=0.05, total_time=0.2, dt=0.002,
                   bc='periodic')
    for policy in ('warn', 'ask', 'raise', 'ignore'):
        with pytest.raises(ValueError, match='agree at x=left and x=right'):
            solve(**problem, on_warning=policy)


def test_preflight_warnings_standalone():
    """The same checks are callable without running anything."""
    grid, _ = make_grid(-5.0, 5.0, 0.05)
    messages, diagnostics = preflight_warnings(
        grid, lambda x, t: -x, lambda x, t: 0.5 + 0 * x,
        dt=0.01, total_time=0.5, method='forward euler', bc_kind='neumann')

    assert len(messages) == 1
    assert diagnostics['stable'] is False
    assert diagnostics['combined_CFL'] == pytest.approx(4.995)
    assert np.isclose(diagnostics['dt_suggested'], 0.01 / 4.995)
