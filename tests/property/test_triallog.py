import json

from hypothesis import given
from hypothesis import strategies as st

from overfitlab import canonical_params_json

SCALARS = (
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False)
)
VALUES = st.recursive(
    SCALARS | st.text(),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(), inner, max_size=3),
    max_leaves=8,
)
PARAMS = st.dictionaries(st.text(), VALUES, max_size=5)


@given(PARAMS)
def test_canonical_json_is_idempotent(params):
    text = canonical_params_json(params)
    assert canonical_params_json(json.loads(text)) == text


@given(PARAMS)
def test_canonical_json_is_ascii_and_compact(params):
    assert canonical_params_json(params).isascii()
