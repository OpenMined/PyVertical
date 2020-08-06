"""This module contains utility functions that expose functionality of PSI."""

from . import client, server


class Server:
    def __init__(self, server_items, fpr=1e-9):
        reveal_intersection = True
        self._server = server.CreateWithNewKey(reveal_intersection)
        self._items = server_items
        self._fpr = fpr

    def process_request(self, request, len_client_items):
        setup = self._server.CreateSetupMessage(
            self._fpr, len_client_items, self._items
        )
        response = self._server.ProcessRequest(request)
        return setup, response


class Client:
    def __init__(self, client_items):
        reveal_intersection = True
        self._client = client.CreateWithNewKey(reveal_intersection)
        self._items = client_items
        self.request = self._client.CreateRequest(client_items)

    def compute_intersection(self, setup, response):
        return self._client.GetIntersection(setup, response)


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
