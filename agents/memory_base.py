# -*- coding:utf-8 -*-
from abc import ABC, abstractmethod


class MemoryBase(ABC):
    @abstractmethod
    def initialize_user(self, user_name: str) -> str: ...

    @abstractmethod
    def chat(self, text: str) -> str: ...

    def finalize_session(self):
        """Called when the session ends. Override only in modules that need it (default: no-op)."""
        pass
