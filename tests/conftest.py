"""Tests are local only; an accidental network call must fail immediately."""
import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not contact Chalk or download models")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import grpc
    from chalkcompute import ConnectClient
    from chalkcompute._remote_call_client import RemoteCallClient
    from chalk.client.client_impl import ChalkAPIClientImpl

    # ChalkClient is a factory; the transport methods belong to its concrete
    # implementation. Patching the factory either fails or misses real calls.
    monkeypatch.setattr(ChalkAPIClientImpl, "_request", forbidden)
    monkeypatch.setattr(ChalkAPIClientImpl, "_get_grpc_client", forbidden)
    monkeypatch.setattr(ConnectClient, "call", forbidden)
    # Compute auth and Function calls use native HTTP/RPC transports which
    # bypass Python sockets. Stop them before credentials or connections load.
    monkeypatch.setattr(ConnectClient, "_ensure_auth", forbidden)
    monkeypatch.setattr(RemoteCallClient, "__init__", forbidden)
    for transport in (grpc, grpc.aio):
        monkeypatch.setattr(transport, "secure_channel", forbidden)
        monkeypatch.setattr(transport, "insecure_channel", forbidden)
