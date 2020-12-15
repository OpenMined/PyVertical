"""This module contains utility functions that expose functionality of PSI."""
from openmined_psi import client
from openmined_psi import server


class Server:
    """
    Class to represent the server in a client/server PSI model.
    """

    def __init__(self, server_items, fpr=1e-9):
        """
        Args:
            server_items (List[str]) : The items provided by the server
            fpr (float) : The false positive ratio
        """

        if len(server_items) == 0:
            raise RuntimeError("Server items cannot be empty")
        reveal_intersection = True
        self._server = server.CreateWithNewKey(reveal_intersection)
        self._items = server_items
        self._fpr = fpr

    def process_request(self, request, len_client_items):
        """
        Return the setup and corresponding response for the client to compute
        the private set intersection.
        Args:
            request (_psi_bindings.PsiProtoRequest): The client request
            len_client_items (int): The length of the client items
        Returns:
            A tuple of (setup, response) with:
            setup (_psi_bindings.PsiProtoServerSetup): The server setup
            response (_psi_bindings.PsiProtoResponse): The server response
        """
        setup = self._server.CreateSetupMessage(
            self._fpr, len_client_items, self._items
        )
        response = self._server.ProcessRequest(request)
        return setup, response


class Client:
    """
    Class to represent the client in a client/server PSI model.
    """

    def __init__(self, client_items):
        """
        Args:
            client_items (List[str]) : The items provided by the client
        """
        if len(client_items) == 0:
            raise RuntimeError("Server items cannot be empty")
        reveal_intersection = True
        self._client = client.CreateWithNewKey(reveal_intersection)
        self._items = client_items
        self.request = self._client.CreateRequest(client_items)

    def compute_intersection(self, setup, response):
        """
        Return the intersection of client and server items.

        Args:
            setup (_psi_bindings.PsiProtoServerSetup): The server setup
            response (_psi_bindings.PsiProtoResponse): The server response
        Returns:
            The intersection set (List[str]) of client and server items
        """
        return sorted(self._client.GetIntersection(setup, response))
