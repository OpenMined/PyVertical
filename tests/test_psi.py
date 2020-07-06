"""
Test code in src/psi
"""
import pytest

from src.psi.util import compute_psi


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            ([str(i) for i in range(10)], [str(i * 2) for i in range(10)]),
            [0, 2, 4, 6, 8],
        ),
        ((["0"], ["0"]), [0]),
        ((["1"], ["2"]), []),
        ((["1"], []), []),
        (([], ["1"]), []),
        (([], []), []),
    ],
)
def test_compute_psi_returns_correct_indices(test_input, expected):
    assert expected == compute_psi(*test_input)
