"""This module contains utility functions that expose functionality of PSI."""

from . import client, server


def compute_psi(client_items, server_items, reveal_intersection=True, fpr=1e-9):
    """Compute the private set intersection of `client_items` and `server_items`.

    Args:
        client_items (List[str]) : The items provided by the client
        server_items (List[str]) : The items provided by the server
        reveal_intersection (bool) : If True, return the common items as a list
        fpr (float) : The false positive ratio

    Returns:
        The cardinality (int) of the intersection set or the intersection
        set (List[str]) itself of client and server items
    """
    c = client.CreateWithNewKey(reveal_intersection)
    s = server.CreateWithNewKey(reveal_intersection)

    setup = s.CreateSetupMessage(fpr, len(client_items), server_items)
    request = c.CreateRequest(client_items)
    resp = s.ProcessRequest(request)

    if reveal_intersection:
        intersection = c.GetIntersection(setup, resp)
    else:
        intersection = c.GetIntersectionSize(setup, resp)

    return intersection
