"""Shared fixtures for the regression suite.

The suite never loads a real model: :class:`FakeScorer` substitutes for
``zero_shot.core.scorer.Scorer`` so prompt building, calibration, softmax, the
classifier, and both interfaces can be exercised deterministically.
"""

from __future__ import annotations

import pytest

from zero_shot.core.models import SequenceScore, TokenScore

CALIBRATION_MARKER = "N/A"

# Every environment variable the config layer reads; cleared per test so an
# ambient ZERO_SHOT_* (e.g. the Docker image sets ZERO_SHOT_CONFIG) can't leak in.
_CONFIG_ENV_VARS = (
    "ZERO_SHOT_CONFIG",
    "ZERO_SHOT_MODEL",
    "ZERO_SHOT_SAVE_TO",
    "ZERO_SHOT_DEVICE",
    "ZERO_SHOT_GPU",
    "ZERO_SHOT_QUANTIZE",
)


class FakeScorer:
    """Deterministic stand-in for ``zero_shot.core.scorer.Scorer``.

    ``logprobs`` maps each scored continuation to its total log-probability.
    ``prior`` maps the same keys to the content-free prior used by the
    calibration pass; the calibration call is detected via ``chat_template``
    (image pass) or a prefix containing the calibration context. Options listed
    in ``omit`` are not scored at all (simulating a scorer that drops an option).
    ``calls`` records every invocation for assertions on the routing.
    """

    device = "cpu"
    multimodal = True

    def __init__(self, logprobs=None, prior=None, omit=()):
        self.logprobs = dict(logprobs or {})
        self.prior = dict(prior or {})
        self.omit = set(omit)
        self.calls: list[dict] = []

    def score_options(
        self,
        prefix_text,
        options,
        *,
        add_leading_space=True,
        use_kv_cache=True,
        image=None,
        chat_template=False,
    ):
        self.calls.append(
            {
                "prefix": prefix_text,
                "options": tuple(options),
                "add_leading_space": add_leading_space,
                "use_kv_cache": use_kv_cache,
                "has_image": image is not None,
                "chat_template": chat_template,
            }
        )
        calibration = chat_template or CALIBRATION_MARKER in prefix_text
        table = self.prior if (calibration and self.prior) else self.logprobs
        results = []
        for option in options:
            if option in self.omit:
                continue
            total = table.get(option, 0.0)
            results.append(
                SequenceScore(
                    option=option,
                    continuation=option,
                    tokens=[TokenScore(token=option, token_id=0, logprob=total)],
                    eos_token="<eos>",
                    eos_logprob=0.0,
                    total_logprob=total,
                )
            )
        return results

    def count_input_tokens(self, prefix_text, image=None):
        return 1


@pytest.fixture(autouse=True)
def _clear_config_env(monkeypatch):
    """Keep config tests independent of ambient ZERO_SHOT_* variables."""
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_scorer(monkeypatch):
    """Install a :class:`FakeScorer` into the classifier; return it for setup."""
    scorer = FakeScorer()
    monkeypatch.setattr("zero_shot.core.classifier.get_scorer", lambda *a, **k: scorer)
    return scorer


@pytest.fixture
def clean_scorer_registry():
    """Ensure the global scorer cache is empty before and after a test."""
    from zero_shot.core import scorer as scorer_module

    scorer_module._SCORERS.clear()
    yield
    scorer_module._SCORERS.clear()


@pytest.fixture
def png_bytes():
    """A tiny valid PNG, for exercising the image endpoints/utilities."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return buffer.getvalue()
