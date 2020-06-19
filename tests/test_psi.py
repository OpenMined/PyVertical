"""
Test code in src/psi
"""

from src.psi.util import compute_psi

def test_compute_psi_returns_correct_indices():
    client_items = [str(i) for i in range(10)]
    server_items = [str(i*2) for i in range(10)]
    assert [0, 2, 4, 6, 8] == compute_psi(client_items, server_items)
