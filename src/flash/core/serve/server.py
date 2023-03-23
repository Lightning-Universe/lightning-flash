import os

from flash.core.serve.interfaces.http import setup_http_app
from flash.core.utilities.imports import _FASTAPI_AVAILABLE, _UVICORN_AVAILABLE

if _UVICORN_AVAILABLE:
    import uvicorn

if _FASTAPI_AVAILABLE:
    from fastapi import FastAPI
else:
    FastAPI = None

FLASH_DISABLE_SERVE = os.getenv("FLASH_DISABLE_SERVE", None)


class ServerMixin:
    """Start a server to serve a composition.

    debug     If the server should be started up in debug mode. By default, False. testing     If the server should
    return the ``app`` instance instead of blocking     the process (via running the ``app`` in ``uvicorn``). This is
    used     when taking advantage of a server ``TestClient``. By default, False
    """

    DEBUG: bool
    TESTING: bool

    def http_app(self) -> "FastAPI":
        return setup_http_app(composition=self, debug=self.DEBUG)

    def serve(self, host: str = "127.0.0.1", port: int = 8000):
        """Start a server to serve a composition.

        Parameters
        ----------
        host
            host address to run the server on
        port
            port number to expose the running server on
        """
        if FLASH_DISABLE_SERVE:
            return

        if not self.TESTING:  # pragma: no cover
            app = self.http_app()
            uvicorn.run(app, host=host, port=port)
        return self.http_app()
