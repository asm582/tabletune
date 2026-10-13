"""The in-engine controller: hold K near table edges, re-fit the table from live
acceptance, and the vLLM plugin that attaches it to the scheduler. See README.md."""

from .controller import DynamicSDController, controller_from_env, extend_prior

__all__ = ["DynamicSDController", "controller_from_env", "extend_prior"]
