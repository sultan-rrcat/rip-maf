"""Small class-introspection helpers shared by contract enforcement."""

from __future__ import annotations


def is_abstract(cls: type) -> bool:
    """True while `cls` still has abstract methods without a concrete override.

    inspect.isabstract() is unusable inside __init_subclass__: ABCMeta sets
    __abstractmethods__ only after type.__new__ returns, and type.__new__ is
    exactly what invokes __init_subclass__. Walking the MRO directly gives the
    same answer at class-definition time, so contract base classes can enforce
    metadata on concrete subclasses while abstract intermediates (the plugin
    contracts themselves) stay exempt.
    """
    pending: set[str] = set()
    # Base -> subclass order: a concrete override closer to `cls` must win
    # over the abstract declaration it replaces.
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if getattr(value, "__isabstractmethod__", False):
                pending.add(name)
            else:
                pending.discard(name)
    return bool(pending)
