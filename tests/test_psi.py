"""
Test code in src/psi
"""
import sys

import pytest

from src.psi.util import Client, Server


@pytest.mark.parametrize(
    "test_input,expected",
    [
        pytest.param(
            ([str(i) for i in range(10)], [str(i * 2) for i in range(10)]),
            [0, 2, 4, 6, 8],
            marks=pytest.mark.xfail(sys.version_info.minor == 8, reason="PSI bug")
        ),
        ((["0"], ["0"]), [0]),
        ((["1"], ["2"]), []),
    ],
)
def test_compute_intersection_returns_correct_indices(test_input, expected):
    client_items = test_input[0]
    server_items = test_input[1]

    client = Client(client_items)
    server = Server(server_items)

    setup, response = server.process_request(client.request, len(client_items))
    intersection = client.compute_intersection(setup, response)

    assert expected == intersection


@pytest.mark.parametrize(
    "test_input,expected", [((["1"], []), []), (([], ["1"]), []), (([], []), []),],
)
def test_compute_intersection_returns_correct_indices_with_empty_items(
    test_input, expected
):
    client_items = test_input[0]
    server_items = test_input[1]

    with pytest.raises(RuntimeError):
        client = Client(client_items)
        server = Server(server_items)
