from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from functools import wraps
from inspect import iscoroutinefunction
from logging import getLogger
from threading import Lock, current_thread
from typing import Any, Literal, Protocol, cast, overload

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QApplication
from vsengine.loops import EventLoop, get_loop

type _CoroutineFunc[**P, R] = Callable[P, Coroutine[Any, Any, R]]
type _Func[**P, R] = Callable[P, R]

_logger = getLogger(__name__)


class QtEventLoop(QObject, EventLoop):
    """Qt event loop adapter for vsview."""

    _invoke = Signal(int)

    def attach(self) -> None:
        self._lock = Lock()
        self._counter = 0
        self._tasks_lock = Lock()
        self._active_tasks = Counter[str]()
        self._pending = dict[int, Callable[[], None]]()
        self._invoke.connect(self._on_invoke)

    def detach(self) -> None:
        self.wait_for_threads(5000)
        self._invoke.disconnect(self._on_invoke)
        self._pending.clear()

    @Slot(int)
    def _on_invoke(self, task_id: int) -> None:
        with self._lock:
            wrapper = self._pending.pop(task_id, None)
        if wrapper:
            wrapper()

    def from_thread[**P, R](self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        """Schedule func to run on the main Qt thread."""
        fut = Future[R]()

        def wrapper() -> None:
            if not fut.set_running_or_notify_cancel():
                return
            try:
                result = func(*args, **kwargs)
            except BaseException as e:
                _logger.debug(e, exc_info=True)
                fut.set_exception(e)
            else:
                fut.set_result(result)

        with self._lock:
            task_id = self._counter
            self._counter += 1
            self._pending[task_id] = wrapper

        self._invoke.emit(task_id)

        return fut

    def to_thread[**P, R](self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        """Run func in Qt's global thread pool."""
        fut = Future[R]()

        def wrapper() -> None:
            with self._tasks_lock:
                self._active_tasks[func.__name__] += 1

            try:
                if not fut.set_running_or_notify_cancel():
                    return
                try:
                    result = func(*args, **kwargs)
                except BaseException as e:
                    _logger.debug(e, exc_info=True)
                    fut.set_exception(e)
                else:
                    fut.set_result(result)
            finally:
                with self._tasks_lock:
                    self._active_tasks[func.__name__] -= 1

        QThreadPool.globalInstance().start(QRunnable.create(wrapper))
        return fut

    def to_thread_named[**P, R](self, name: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        """Run func in Qt's global thread pool with a custom thread name."""
        fut = Future[R]()

        def wrapper() -> None:
            with self._tasks_lock:
                self._active_tasks[name] += 1

            try:
                current_thread().name = name
                if not fut.set_running_or_notify_cancel():
                    return
                try:
                    result = func(*args, **kwargs)
                except BaseException as e:
                    _logger.debug(e, exc_info=True)
                    fut.set_exception(e)
                else:
                    fut.set_result(result)
            finally:
                with self._tasks_lock:
                    self._active_tasks[name] -= 1

        QThreadPool.globalInstance().start(QRunnable.create(wrapper))
        return fut

    def wait_for_threads(self, timeout_ms: int = 500) -> None:
        """
        Wait for background threads to finish.
        If called from a background thread, it ignores itself in the count.
        """
        _logger.debug("Calling wait_for_threads...")

        if not (app := QApplication.instance()):
            _logger.warning("No QApplication instance found")
            return

        is_main_thread = QThread.currentThread() == app.thread()
        current_name = current_thread().name

        with self._tasks_lock:
            target_count = int(self._active_tasks.get(current_name, 0) > 0)

        pool = QThreadPool.globalInstance()

        for _ in range(max(1, timeout_ms // 10)):
            count = pool.activeThreadCount()

            if count <= target_count:
                _logger.debug("Target thread count reached (%s), breaking loop.", count)
                break

            with self._tasks_lock:
                running = [k for k, v in self._active_tasks.items() if v > 0]

            _logger.debug("Active threads: %s (Target: %s)", count, target_count)
            _logger.debug("Running tasks: %r", running)

            _logger.debug("Waiting 10 ms...")
            pool.waitForDone(10)

            if is_main_thread:
                QApplication.processEvents()
        else:
            _logger.warning(
                "Timeout of %dms reached while waiting for threads. "  # No fmt
                "Proceeding while %d thread(s) are still active.",
                timeout_ms,
                pool.activeThreadCount(),
            )

        if is_main_thread:
            # Final flush to process any signals from threads that just finished
            QApplication.processEvents()

        with self._tasks_lock:
            _logger.debug("Final active threads count: %s", pool.activeThreadCount())
            _logger.debug("Remaining thread(s): %s", [k for k, v in self._active_tasks.items() if v > 0])


class _DecoratorFuture(Protocol):
    """Protocol for decorators that wrap a function and return a Future."""

    @overload
    def __call__[**P, R](self, func: _CoroutineFunc[P, R]) -> Callable[P, Future[R]]: ...
    @overload
    def __call__[**P, R](self, func: _Func[P, R]) -> Callable[P, Future[R]]: ...


class _DecoratorDirect(Protocol):
    """Protocol for decorators that wrap a function and return the result directly."""

    @overload
    def __call__[**P, R](self, func: _CoroutineFunc[P, R]) -> Callable[P, R]: ...
    @overload
    def __call__[**P, R](self, func: _Func[P, R]) -> Callable[P, R]: ...


@overload
def run_in_loop[**P, R](func: _CoroutineFunc[P, R]) -> Callable[P, Future[R]]: ...
@overload
def run_in_loop[**P, R](func: _Func[P, R]) -> Callable[P, Future[R]]: ...
@overload
def run_in_loop(*, return_future: Literal[True]) -> _DecoratorFuture: ...
@overload
def run_in_loop(*, return_future: Literal[False]) -> _DecoratorDirect: ...
@overload
def run_in_loop(*, return_future: bool) -> _DecoratorFuture | _DecoratorDirect: ...
def run_in_loop(func: Any = None, *, return_future: bool = True) -> Any:
    """
    Decorator. Executes the decorated function within the `QtEventLoop` (Main Thread).

    Args:
        func: The function to wrap (when used as `@run_in_loop` without parens)
        return_future: If False, blocks and returns R directly.

    Returns:
        A future object or the result directly, depending on return_future.

    Usage:
    ```python
    @run_in_loop
    def my_func(): ...


    @run_in_loop(return_future=False)
    def my_blocking_func(): ...
    ```
    """

    def decorator(fn: Any) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = cast(QtEventLoop, get_loop())

            if iscoroutinefunction(fn):

                def run_coro() -> Any:
                    import asyncio

                    coro = fn(*args, **kwargs)
                    try:
                        return asyncio.run(coro)
                    except RuntimeError:
                        return asyncio.run_coroutine_threadsafe(coro, asyncio.get_running_loop()).result()

                fut = loop.from_thread(run_coro)
            else:
                # Delegate to from_thread to marshal execution to the main loop
                fut = loop.from_thread(fn, *args, **kwargs)

            return fut if return_future else fut.result()

        return wrapper

    return decorator if func is None else decorator(func)


@overload
def run_in_background[**P, R](func: _CoroutineFunc[P, R]) -> Callable[P, Future[R]]: ...
@overload
def run_in_background[**P, R](func: _Func[P, R]) -> Callable[P, Future[R]]: ...
@overload
def run_in_background(*, name: str) -> _DecoratorFuture: ...
def run_in_background(func: Any = None, *, name: str | None = None) -> Any:
    """
    Executes the decorated function in a background thread (via QThreadPool)
    using the `QtEventLoop`'s `to_thread` logic.

    Args:
        func: The function to wrap (when used as `@run_in_background` without parens)
        name: Optional thread name for logging (when used as `@run_in_background(name="...")`)

    Returns:
        A future object representing the result of the execution.

    Usage:
    ```python
    @run_in_background
    def my_func(): ...


    @run_in_background(name="MyWorker")
    def my_named_func(): ...
    ```
    """

    def decorator(fn: Any) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = cast(QtEventLoop, get_loop())

            if iscoroutinefunction(fn):

                def run_coro() -> Any:
                    import asyncio

                    coro = fn(*args, **kwargs)
                    try:
                        return asyncio.run(coro)
                    except RuntimeError:
                        return asyncio.run_coroutine_threadsafe(coro, asyncio.get_running_loop()).result()

                return loop.to_thread(run_coro) if name is None else loop.to_thread_named(name, run_coro)

            return (
                loop.to_thread(fn, *args, **kwargs) if name is None else loop.to_thread_named(name, fn, *args, **kwargs)
            )

        return wrapper

    return decorator if func is None else decorator(func)
