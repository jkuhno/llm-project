"""Define the configurable parameters for the agent.
Borrowed directly from langgraph server react-agent-python template.
"""

from __future__ import annotations

import uuid

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config # type: ignore


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""
    
    thread_id: str = field(
        default="abc123",
        metadata={
            "description": "The thread ID for the current conversation."
        },
    )

    user_id: str = field(
        default="user_1",
        metadata={
            "description": "The user ID for the current session."
        },
    )

    mem_key: int = field(
        default=uuid.uuid4(),
        metadata={
            "description": "The memory key for the current session."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
