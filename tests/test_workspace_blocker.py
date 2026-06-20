from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QObject
from pytest_mock import MockerFixture

from vsview.app.plugins.api import PluginAPI, WorkspaceBlocker


def create_api(mocker: MockerFixture) -> Any:
    mock_workspace = mocker.MagicMock()
    return PluginAPI(mock_workspace)


def test_workspace_blocker_init(mocker: MockerFixture) -> None:
    api = create_api(mocker)
    caller = QObject()

    # Create blocker with caller
    blocker = api.blocker(caller)
    assert isinstance(blocker, WorkspaceBlocker)
    assert not getattr(blocker, "locked")

    # Create blocker without caller
    blocker2 = api.blocker()
    assert isinstance(blocker2, WorkspaceBlocker)
    assert not getattr(blocker2, "locked")


def test_workspace_blocker_acquire_release(mocker: MockerFixture) -> None:
    api = create_api(mocker)
    caller = QObject()
    blocker = api.blocker(caller)

    assert getattr(api, "busy") is False

    # Acquire
    assert blocker.acquire() is True
    assert getattr(blocker, "locked") is True
    assert getattr(api, "busy") is True

    # Release
    blocker.release()
    assert getattr(blocker, "locked") is False
    assert getattr(api, "busy") is False


def test_workspace_blocker_acquire_nonblocking(mocker: MockerFixture) -> None:
    api = create_api(mocker)
    blocker = api.blocker()

    assert blocker.acquire() is True
    assert getattr(blocker, "locked") is True
    assert getattr(api, "busy") is True

    # Non-blocking acquire on already acquired lock should fail
    assert blocker.acquire(block=False) is False

    blocker.release()
    assert getattr(blocker, "locked") is False
    assert getattr(api, "busy") is False

    # Acquire again should succeed
    assert blocker.acquire(block=False) is True
    assert getattr(blocker, "locked") is True
    assert getattr(api, "busy") is True
    blocker.release()


def test_workspace_blocker_context_manager(mocker: MockerFixture) -> None:
    api = create_api(mocker)
    caller = QObject()
    blocker = api.blocker(caller)

    assert getattr(api, "busy") is False

    with blocker:
        assert getattr(blocker, "locked") is True
        assert getattr(api, "busy") is True

    assert getattr(blocker, "locked") is False
    assert getattr(api, "busy") is False


def test_workspace_blocker_threads(mocker: MockerFixture) -> None:
    api = create_api(mocker)
    blocker = api.blocker()

    # Acquire in main thread
    assert blocker.acquire() is True
    assert getattr(blocker, "locked") is True

    # Attempt to acquire in background thread with timeout
    results = []

    def worker() -> None:
        success = blocker.acquire(block=True, timeout=0.1)
        results.append(success)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert results == [False]

    # Release main thread
    blocker.release()

    # Now background thread should be able to acquire
    t2 = threading.Thread(target=worker)
    t2.start()
    t2.join()
    assert results == [False, True]


def test_api_blocker_factory(mocker: MockerFixture) -> None:
    mock_api: Any = mocker.MagicMock(spec=PluginAPI)
    caller: Any = QObject()

    blocker = PluginAPI.blocker(mock_api, caller)
    assert isinstance(blocker, WorkspaceBlocker)
