"""Secrets manager backed by the OS keyring."""

from __future__ import annotations

import json
from contextlib import suppress
from importlib.util import find_spec
from logging import getLogger
from types import ModuleType
from typing import TYPE_CHECKING, Any

from jetpytools import Singleton, cachedproperty, inject_self

if TYPE_CHECKING:
    from keyring.credentials import Credential

logger = getLogger(__name__)


class SecretsError(RuntimeError):
    """Raised when secure secret storage is unavailable or fails."""


class SecretsManager(Singleton):
    """Manage secure storage of plugin/application secrets."""

    prefix_service_name = "jet.vsview"

    if TYPE_CHECKING:
        import keyring
        from keyring.errors import KeyringError, PasswordDeleteError

    else:

        @cachedproperty
        def keyring(self) -> ModuleType:
            if find_spec("keyring"):
                import keyring

                return keyring

            raise SecretsError("The 'keyring' dependency is not installed.")

        @cachedproperty
        def KeyringError(self) -> Exception:
            if find_spec("keyring"):
                from keyring.errors import KeyringError

                return KeyringError

            raise SecretsError("The 'keyring' dependency is not installed.")

        @cachedproperty
        def PasswordDeleteError(self) -> Exception:
            if find_spec("keyring"):
                from keyring.errors import PasswordDeleteError

                return PasswordDeleteError

            raise SecretsError("The 'keyring' dependency is not installed.")

    @inject_self
    def get(self, namespace: str, key: str) -> str | None:
        try:
            return self.keyring.get_password(f"{self.prefix_service_name}.{namespace}", key)
        except self.KeyringError as exc:
            raise SecretsError("Failed to read secret from keyring backend.") from exc

    @inject_self
    def set(self, namespace: str, key: str, value: str) -> None:
        if not key:
            logger.warning("Empty usernames are prohibited.")
            return

        # https://github.com/jaraco/keyring/issues/545
        with suppress(SecretsError):
            self.delete(namespace, key)

        try:
            self.keyring.set_password(f"{self.prefix_service_name}.{namespace}", key, value)
        except self.KeyringError as exc:
            raise SecretsError("Failed to store secret in keyring backend.") from exc

    @inject_self
    def delete(self, namespace: str, key: str) -> None:
        try:
            self.keyring.delete_password(f"{self.prefix_service_name}.{namespace}", key)
        except self.PasswordDeleteError:
            logger.debug("No secret exists for %s:%s", namespace, key)
        except self.KeyringError as exc:
            raise SecretsError("Failed to delete secret from keyring backend.") from exc

    @inject_self
    def get_credential(self, namespace: str) -> Credential | None:
        try:
            return self.keyring.get_credential(f"{self.prefix_service_name}.{namespace}", None)
        except Exception as exc:
            raise SecretsError("Failed to get credential from keyring backend.") from exc

    @inject_self
    def get_json(self, namespace: str, key: str) -> Any | None:
        if (raw := self.get(namespace, key)) is None:
            return None

        try:
            return json.loads(raw)
        except Exception as exc:
            raise SecretsError(f"Secret for {namespace}:{key} is not valid JSON.") from exc

    @inject_self
    def set_json(self, namespace: str, key: str, value: Any) -> None:
        self.set(namespace, key, json.dumps(value, separators=(",", ":")))
