"""把 selftest.py 的每一项检查作为独立的 pytest 用例跑。

复用 `selftest.CHECKS` 而不是重写一遍断言:两套断言各写一遍必然会走样。
"""

import pytest

from selftest import CHECKS


@pytest.mark.parametrize("name,fn", CHECKS, ids=[n for n, _ in CHECKS])
def test_check(name, fn):
    ok, detail = fn()
    assert ok, f"{name}: {detail}"
