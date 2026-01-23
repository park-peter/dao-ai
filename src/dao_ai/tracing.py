"""Helpers for MLflow tracing integration."""

from __future__ import annotations


def patch_mlflow_tracing_for_parent_command() -> None:
    """Patch MLflow tracing to ignore ParentCommand exceptions.

    ParentCommand is an internal LangGraph control flow mechanism, not a real error.
    This patch prevents MLflow from recording it as an exception in traces.
    """
    try:
        import inspect

        from langgraph.errors import ParentCommand
        from mlflow.tracing import fluent as mlflow_fluent
        import opentelemetry.trace as ot_trace
        try:
            from opentelemetry.sdk.trace import Span as OtelSpan
        except Exception:
            OtelSpan = None
    except Exception:
        return

    def _patch_span_class(span_cls) -> None:
        original_record_exception = getattr(span_cls, "record_exception", None)
        original_set_status = getattr(span_cls, "set_status", None)

        if original_record_exception and not getattr(
            original_record_exception, "_dao_ai_patched", False
        ):

            def _patched_record_exception(self, exception, *args, **kwargs):
                if isinstance(exception, ParentCommand):
                    setattr(self, "_dao_ai_parent_command", True)
                    return None
                return original_record_exception(self, exception, *args, **kwargs)

            _patched_record_exception._dao_ai_patched = True
            span_cls.record_exception = _patched_record_exception

        if original_set_status and not getattr(
            original_set_status, "_dao_ai_patched", False
        ):

            def _patched_set_status(self, status, *args, **kwargs):
                if getattr(self, "_dao_ai_parent_command", False):
                    try:
                        status_code = getattr(status, "status_code", None)
                        if status_code and str(status_code).endswith("ERROR"):
                            return None
                    except Exception:
                        pass
                return original_set_status(self, status, *args, **kwargs)

            _patched_set_status._dao_ai_patched = True
            span_cls.set_status = _patched_set_status

    span_cls = OtelSpan or getattr(ot_trace, "Span", None)
    if span_cls is not None:
        _patch_span_class(span_cls)

    try:
        import opentelemetry.trace.span as ot_span

        for cls_name in ("Span", "ProxySpan", "NonRecordingSpan"):
            cls = getattr(ot_span, cls_name, None)
            if cls is not None:
                _patch_span_class(cls)
    except Exception:
        pass

    # Patch LangChain tracer error handlers to skip ParentCommand
    try:
        import mlflow.langchain.langchain_tracer as langchain_tracer
    except Exception:
        langchain_tracer = None

    if langchain_tracer is not None:

        def _patch_error_handler(cls, method_name: str) -> None:
            original = getattr(cls, method_name, None)
            if original is None or getattr(original, "_dao_ai_patched", False):
                return

            def _wrapped(self, error, *args, **kwargs):
                if isinstance(error, ParentCommand):
                    return None
                return original(self, error, *args, **kwargs)

            _wrapped._dao_ai_patched = True
            setattr(cls, method_name, _wrapped)

        for cls_name in ("BaseCallbackHandler", "MlflowLangchainTracer"):
            cls = getattr(langchain_tracer, cls_name, None)
            if cls is None:
                continue
            for method_name in (
                "on_chain_error",
                "on_llm_error",
                "on_retriever_error",
                "on_tool_error",
            ):
                _patch_error_handler(cls, method_name)

    # Patch _wrap_function to intercept _WrappingContext.__exit__
    if hasattr(mlflow_fluent, "_wrap_function") and not getattr(
        mlflow_fluent._wrap_function, "_dao_ai_patched", False
    ):
        original_wrap_function = mlflow_fluent._wrap_function

        def _patched_wrap_function(*args, **kwargs):
            wrapped = original_wrap_function(*args, **kwargs)
            try:
                freevars = getattr(wrapped, "__code__", None)
                closure = wrapped.__closure__ or []
                names = list(freevars.co_freevars) if freevars else []
                for name, cell in zip(names, closure):
                    if name != "_WrappingContext":
                        continue
                    try:
                        context_cls = cell.cell_contents
                    except Exception:
                        continue
                    for method_name in ("__exit__", "__aexit__"):
                        original = getattr(context_cls, method_name, None)
                        if original is None or getattr(
                            original, "_dao_ai_patched", False
                        ):
                            continue
                        if inspect.iscoroutinefunction(original):

                            async def _patched_exit(
                                self, exc_type, exc, traceback, _orig=original
                            ):
                                if exc_type is not None:
                                    try:
                                        is_parent = issubclass(
                                            exc_type, ParentCommand
                                        )
                                    except Exception:
                                        is_parent = False
                                    if is_parent:
                                        await _orig(self, None, None, None)
                                        return False
                                return await _orig(self, exc_type, exc, traceback)

                        else:

                            def _patched_exit(
                                self, exc_type, exc, traceback, _orig=original
                            ):
                                if exc_type is not None:
                                    try:
                                        is_parent = issubclass(
                                            exc_type, ParentCommand
                                        )
                                    except Exception:
                                        is_parent = False
                                    if is_parent:
                                        _orig(self, None, None, None)
                                        return False
                                return _orig(self, exc_type, exc, traceback)

                        _patched_exit._dao_ai_patched = True
                        setattr(context_cls, method_name, _patched_exit)
            except Exception:
                pass
            return wrapped

        _patched_wrap_function._dao_ai_patched = True
        mlflow_fluent._wrap_function = _patched_wrap_function
