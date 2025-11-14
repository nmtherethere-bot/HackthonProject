from copy import deepcopy
from typing import Any, Mapping, MutableMapping


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively update mapping 'base' with 'override'.

    - Dicts are merged recursively.
    - Other types (scalars/lists) are replaced by override.
    - Returns the mutated base for convenience.
    """
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, Mapping)
        ):
            deep_update(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base