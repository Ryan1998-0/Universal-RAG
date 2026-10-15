"""Count actual ASGI bytes, including requests without Content-Length."""

from starlette.exceptions import HTTPException


class RequestBodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        observed = 0

        async def limited_receive():
            nonlocal observed
            message = await receive()
            if message["type"] == "http.request":
                observed += len(message.get("body", b""))
                if observed > self.max_bytes:
                    raise HTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send)
