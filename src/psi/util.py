"""This module contains utility functions that expose functionality of PSI."""

from . import client, server


def compute_psi(client_items, server_items, fpr=1e-9):
    """Compute the private set intersection of `client_items` and `server_items`.

    Args:
        client_items (List[str]) : The items provided by the client
        server_items (List[str]) : The items provided by the server
        fpr (float) : The false positive ratio

    Returns:
        The intersection set (List[str]) of client and server items
    """
    if len(client_items) == 0 or len(server_items) == 0:
        return []

    c = client.CreateWithNewKey(True)
    s = server.CreateWithNewKey(True)

    setup = s.CreateSetupMessage(fpr, len(client_items), server_items)
    request = c.CreateRequest(client_items)
    resp = s.ProcessRequest(request)

    return c.GetIntersection(setup, resp)