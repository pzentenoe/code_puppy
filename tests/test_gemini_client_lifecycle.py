"""Regression tests: the Gemini model must use the shared client factory.

Background: the vanilla ``gemini`` model type built its own bare
``httpx.AsyncClient(timeout=180)``, which opted out of everything the shared
factory provides -- retries, proxy support, ``SSL_CERT_FILE`` trust and the
http2 setting -- while the ``custom_gemini`` path got all of it. The client
it created was also never closed: ``_close_client`` had no caller.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from code_puppy import gemini_model
from code_puppy.gemini_model import GeminiModel


def _model(http_client=None) -> GeminiModel:
    return GeminiModel(model_name="gemini-test", api_key="key", http_client=http_client)


class TestSharedClientFactory:
    async def test_owned_client_comes_from_the_shared_factory(self):
        """No more bare httpx.AsyncClient: one way to build a client."""
        model = _model()

        with patch(
            "code_puppy.http_utils.create_async_client", return_value=object()
        ) as factory:
            client = await model._get_client()

        factory.assert_called_once()
        assert client is factory.return_value

    async def test_injected_client_is_used_as_is(self):
        """An injected client is never replaced."""
        injected = httpx.AsyncClient()
        model = _model(http_client=injected)

        client = await model._get_client()

        assert client is injected


class TestClientTeardown:
    async def test_owned_client_is_closed_on_exit(self):
        """A client the model created itself must not outlive it."""
        model = _model()
        owned = AsyncMock(spec=httpx.AsyncClient)
        model._http_client = owned
        model._owns_client = True

        await model.__aexit__(None, None, None)

        owned.aclose.assert_awaited_once()
        assert model._http_client is None

    async def test_injected_client_is_not_closed(self):
        """The model did not create it, so it must not close it."""
        model = _model(http_client=AsyncMock(spec=httpx.AsyncClient))
        model._owns_client = False

        await model.__aexit__(None, None, None)

        model._http_client.aclose.assert_not_awaited()


class TestModelFactoryInjectsTheClient:
    def test_gemini_branch_injects_a_shared_client(self):
        """The vanilla gemini type must go through the shared factory."""
        from code_puppy import model_factory

        model_config = {"name": "gemini-2.5-pro", "type": "gemini"}
        fake_client = object()

        with (
            patch.object(model_factory, "get_api_key", return_value="key"),
            patch.object(
                model_factory, "create_async_client", return_value=fake_client
            ) as factory,
            patch.object(model_factory, "get_cert_bundle_path", return_value=None),
        ):
            model = model_factory.ModelFactory.get_model(
                "gemini-2.5-pro", {model_config["name"]: model_config}
            )

        factory.assert_called_once()
        assert model is not None
        assert model._http_client is fake_client
        # An injected client is not owned by the model.
        assert model._owns_client is False


def test_module_exposes_the_close_helper():
    """The teardown helper is part of the model's contract."""
    assert callable(gemini_model.GeminiModel._close_client)
