# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming-safe ASGI trace milestones and HTTP clock-exchange samples."""
from vllm.pd_trace import clock_sample, get_trace


class TraceMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        trace = get_trace()
        if trace is None or scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", ()))
        fields = {"request_id": headers.get(b"x-request-id", b"").decode(),
                  "user_request_id": headers.get(b"x-client-request-id", b"").decode(),
                  "path": scope.get("path"), "role": "api"}
        ingress_clock = clock_sample()
        ingress = trace.emit("api_ingress", clock=ingress_clock, **fields)
        parent = headers.get(b"x-pd-trace-parent-id")
        if parent:
            trace.emit("http_dependency", dependencies=[parent.decode()],
                       ingress_event_id=ingress, **fields)
        first = True

        async def traced_send(message):
            nonlocal first
            if message["type"] == "http.response.start":
                sample = clock_sample()
                trace.emit("api_headers", status=message["status"],
                           dependencies=[ingress], clock=sample, **fields)
                message = {**message, "headers": [*message.get("headers", ()),
                    (b"x-pd-trace-ingress-wall-ns", str(ingress_clock["wall_ns"]).encode()),
                    (b"x-pd-trace-headers-wall-ns", str(sample["wall_ns"]).encode()),
                    (b"x-pd-trace-process-id", trace.process_id.encode())]}
            elif message["type"] == "http.response.body":
                final = not message.get("more_body", False)
                if trace.full_detail or first or final:
                    trace.emit("api_stream_chunk", first=first, final=final,
                               byte_count=len(message.get("body", b"")), **fields)
                first = False
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except BaseException:
            trace.emit("api_failed", **fields)
            raise
