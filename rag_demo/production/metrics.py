from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ProductionMetrics:
    def __init__(self):
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "rag_http_requests_total",
            "HTTP requests handled by the RAG API.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "rag_http_request_duration_seconds",
            "End-to-end HTTP request duration.",
            ("method", "route"),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 180),
            registry=self.registry,
        )
        self.http_inflight = Gauge(
            "rag_http_requests_inflight",
            "HTTP requests currently being handled.",
            registry=self.registry,
        )
        self.generation_waiting = Gauge(
            "rag_generation_waiting",
            "Requests waiting for a generation slot.",
            registry=self.registry,
        )
        self.generation_active = Gauge(
            "rag_generation_active",
            "Generation requests currently running.",
            registry=self.registry,
        )
        self.pipeline_duration = Histogram(
            "rag_pipeline_duration_seconds",
            "Canonical pipeline stage duration.",
            ("stage",),
            buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 180),
            registry=self.registry,
        )
        self.api_errors = Counter(
            "rag_api_errors_total",
            "Safe API errors returned to clients.",
            ("code",),
            registry=self.registry,
        )

    def observe_pipeline_timings(self, timings: dict) -> None:
        for stage, milliseconds in dict(timings or {}).items():
            try:
                seconds = max(0.0, float(milliseconds) / 1000.0)
            except (TypeError, ValueError):
                continue
            self.pipeline_duration.labels(stage=str(stage)).observe(seconds)


def route_label(scope: dict) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path or "unmatched")
