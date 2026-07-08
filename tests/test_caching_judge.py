"""CachingJudge: memoise deterministic judge calls, bypass when sampling."""

import pytest

from ragas_lingua import CachingJudge


class _Counter:
    """Minimal Judge that counts calls and returns a fixed object."""

    model = "test-model"

    def __init__(self, value, temperature=0.0):
        self.value = value
        self.temperature = temperature
        self.calls = 0

    def structured(self, *, system, user, schema, tool_name="record"):
        self.calls += 1
        return dict(self.value)


def _call(judge, user="u"):
    return judge.structured(system="s", user=user, schema={"type": "object"})


def test_identical_call_is_served_from_cache():
    inner = _Counter({"score": 1})
    cj = CachingJudge(inner)
    r1 = _call(cj)
    r2 = _call(cj)
    assert inner.calls == 1  # second request never reached the inner judge
    assert r1 == r2 == {"score": 1}
    assert cj.stats() == {"hits": 1, "misses": 1, "size": 1}


def test_different_prompt_is_a_miss():
    inner = _Counter({"score": 1})
    cj = CachingJudge(inner)
    _call(cj, user="one")
    _call(cj, user="two")
    assert inner.calls == 2
    assert cj.stats()["size"] == 2


def test_mutating_the_returned_dict_does_not_poison_the_cache():
    inner = _Counter({"score": 1})
    cj = CachingJudge(inner)
    r1 = _call(cj)
    r1["injected"] = 99
    r2 = _call(cj)
    assert "injected" not in r2


def test_temperature_above_zero_bypasses_cache_and_warns():
    inner = _Counter({"score": 1}, temperature=0.7)
    with pytest.warns(UserWarning, match="temperature"):
        cj = CachingJudge(inner)
    _call(cj)
    _call(cj)
    assert inner.calls == 2  # no caching while sampling
    assert cj.temperature == 0.7  # stays transparent to the confidence guard


def test_persists_to_disk_and_reloads(tmp_path):
    path = tmp_path / "cache.json"
    inner1 = _Counter({"score": 0.5})
    cj1 = CachingJudge(inner1, path=path)
    _call(cj1)
    cj1.save()
    assert path.exists()

    inner2 = _Counter({"score": 0.5})
    cj2 = CachingJudge(inner2, path=path)
    result = _call(cj2)
    assert inner2.calls == 0  # served from the reloaded disk cache
    assert result == {"score": 0.5}


def test_context_manager_saves_on_exit(tmp_path):
    path = tmp_path / "cache.json"
    inner1 = _Counter({"ok": True})
    with CachingJudge(inner1, path=path) as cj:
        _call(cj)
    assert path.exists()

    inner2 = _Counter({"ok": True})
    cj2 = CachingJudge(inner2, path=path)
    _call(cj2)
    assert inner2.calls == 0


def test_save_without_path_raises():
    with pytest.raises(ValueError, match="no path"):
        CachingJudge(_Counter({})).save()
