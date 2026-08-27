"""Which platform this process is running on, where that changes what the code may do.

``DATABRICKS_APP_PORT`` is injected by the Apps runtime and by nothing else, so it is the
only signal used. Softer inferences get it wrong: OAuth variables are also present on a
laptop with the SDK configured, and a local run that believes it is an App loses
functionality no platform limit applies to.

Nothing here branches behaviour on its own. A caller pairs it with an explicit config
override, so the platform decides the default and the operator keeps the last word.
"""

import os


def running_as_databricks_app() -> bool:
    """True when this process was started by the Databricks Apps runtime."""
    return bool(str(os.getenv("DATABRICKS_APP_PORT") or "").strip())


def resolve_mode(configured, *, platform_default: bool) -> bool:
    """Read an ``auto`` / on / off tri-state, defaulting to the platform's answer.

    ``auto``, absent and unparseable all mean "let the platform decide"; anything else is
    the operator overriding it in one direction. Unparseable folds into ``auto`` rather
    than being refused because these switches are read at boot with no operator watching,
    so a typo must not stop the server coming up. The caller logs it and
    :func:`is_recognised_mode` reports it.
    """
    text = str(configured if configured is not None else "").strip().lower()
    if text in ("off", "false", "no", "disabled", "0"):
        return False
    if text in ("on", "true", "yes", "enabled", "1"):
        return True
    return platform_default


def is_recognised_mode(configured) -> bool:
    """Whether ``configured`` was a value :func:`resolve_mode` understood.

    Separate from the resolution so a typo is reported while still resolving to a working
    default. Reading ``enabld`` as ``auto`` silently turns a deliberate override into a
    mystery.
    """
    text = str(configured if configured is not None else "").strip().lower()
    return text in (
        "",
        "auto",
        "off",
        "false",
        "no",
        "disabled",
        "0",
        "on",
        "true",
        "yes",
        "enabled",
        "1",
    )
