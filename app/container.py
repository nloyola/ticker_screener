from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class Provider:
    def get(self) -> Any:  # override in subclasses
        raise NotImplementedError


class Value(Provider):
    def __init__(self, value: Any):
        self._value = value

    def get(self) -> Any:
        return self._value


class Singleton(Provider):
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance = _UNSET

    def get(self):
        if self._instance is _UNSET:
            self._instance = self._factory()
        return self._instance


class Factory(Provider):
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory

    def get(self):
        return self._factory()


_UNSET = object()


class Container:
    def __init__(self):
        self._providers: dict[str, Provider] = {}
        self._overrides: list[dict[str, Provider]] = []

    def register(self, name: str, provider: Provider):
        self._providers[name] = provider

    def __getattr__(self, name: str) -> Any:
        prov = self._providers.get(name)
        if prov is None:
            raise AttributeError(name)
        return prov.get()

    @contextmanager
    def override(self, **providers: Provider):
        # push
        self._overrides.append(providers)
        prev: dict[str, Provider] = {}
        try:
            for k, p in providers.items():
                prev[k] = self._providers.get(k)
                self._providers[k] = p
            yield
        finally:
            # pop
            for k, old in prev.items():
                if old is None:
                    self._providers.pop(k, None)
                else:
                    self._providers[k] = old
            self._overrides.pop()
