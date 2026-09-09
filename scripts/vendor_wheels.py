#!/usr/bin/env python3
"""Build `vendor/`: the wheel closure a no-egress Databricks Apps builder installs from.

    python scripts/vendor_wheels.py          # rebuild vendor/ from requirements.txt
    python scripts/vendor_wheels.py --check  # verify the existing vendor/ still fits

Run it after every `requirements.txt` change, or the App installs the previous closure —
`PIP_NO_INDEX` means pip cannot notice that a pin moved. `app.yaml` carries the measurements
behind every constant here; this file is only the arithmetic.

Two rules it exists to enforce, both of which fail as an opaque deploy error otherwise:

  * NOTHING MAY EXCEED 10,485,760 BYTES. That is the per-file limit on workspace-file
    import, and `bundle deploy` uploads `vendor/` through it. An oversized wheel does not
    warn — the file is refused and pip then reports the requirement as unsatisfiable.
  * A WHEEL THE BASE IMAGE ALREADY SATISFIES IS DROPPED, not shipped. numpy and pandas are
    the only two that matter: each is over the cap and each is pre-installed, so pip skips
    the requirement entirely. Dropping them is what makes the rest fit.

The output is deliberately untracked. 37.5 MB of linux/cp311 wheels would outlive the
stopgap in git history, and `bundle deploy` syncs the working tree rather than HEAD.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "vendor"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# The workspace-file import limit, verbatim from the API's own refusal:
# `exceeded max size (10485760 bytes)`.
MAX_FILE_BYTES = 10 * 1024 * 1024

# The Apps runtime, measured by deploying a script that enumerated
# `importlib.metadata.distributions()`: Python 3.11.15, 132 distributions.
TARGET_PYTHON = "3.11"
TARGET_PLATFORM = "manylinux2014_x86_64"

# Pre-installed in the base image at versions that satisfy our pins (numpy 1.26.4 against a
# bare `numpy`, pandas 2.2.3 against `pandas~=2.2.2`), so pip never fetches them. Both are
# over the cap, which is the only reason this list is needed at all. If a base-image bump
# drops one, the deploy fails with `No matching distribution` — visible, not silent.
BASE_IMAGE_SATISFIES = ("numpy", "pandas")


def _distribution_name(wheel: str) -> str:
    return wheel.split("-", 1)[0].replace("_", "-").lower()


def _download(dest: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(dest),
            "--platform",
            TARGET_PLATFORM,
            "--python-version",
            TARGET_PYTHON,
            "--only-binary=:all:",
            "-r",
            str(REQUIREMENTS),
        ],
        check=True,
    )


def _installed_versions() -> dict[str, str]:
    from importlib import metadata

    out = {}
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").replace("_", "-").lower()
        if name:
            out[name] = dist.version
    return out


def _report_version_drift(wheels: list[Path]) -> None:
    """Name every wheel whose version is not the one the suite ran against.

    `pip download --platform` resolves fresh against the index and ignores this environment,
    so a requirement with no upper bound picks up whatever released since — a package
    generation nothing here has executed. That is not hypothetical: `openai>=1.50` resolved
    to 3.6.0, which vendors `httpx_aiohttp` and reads an `aiohttp` attribute added after our
    pin, so the App crashed at import while the suite stayed green. A drift line is not an
    error (the local env is Python 3.10 and holds dev-only packages), but every deploy
    surprise of that shape shows up here first.
    """
    installed = _installed_versions()
    drift = []
    for wheel in wheels:
        name, _, rest = wheel.name.partition("-")
        version = rest.split("-", 1)[0]
        local = installed.get(name.replace("_", "-").lower())
        if local and local != version:
            drift.append((name, local, version))
    if not drift:
        return
    print(f"\n{len(drift)} wheels differ from the version installed here:")
    for name, local, vendored in sorted(drift):
        print(f"  {name:32} local {local:<14} vendored {vendored}")
    print(
        "Each is a version the suite has not executed. Add an upper bound in "
        "requirements.txt for any that matters, then rebuild."
    )


def _report(wheels: list[Path]) -> int:
    oversized = [w for w in wheels if w.stat().st_size > MAX_FILE_BYTES]
    total = sum(w.stat().st_size for w in wheels)
    print(f"\n{len(wheels)} wheels, {total / 1e6:.1f} MB in {VENDOR.relative_to(REPO_ROOT)}/")
    if oversized:
        print(f"\nREFUSED — {len(oversized)} over the {MAX_FILE_BYTES}-byte import cap:")
        for w in sorted(oversized, key=lambda p: -p.stat().st_size):
            print(f"  {w.stat().st_size / 1e6:7.2f} MB  {w.name}")
        print(
            "\nEach of these has to go: drop the requirement that pulls it (see the "
            "snowflake note in requirements.txt), or find a smaller pin. Shipping it "
            "fails the deploy with a message about the requirement, not the file."
        )
        return 1
    largest = max(wheels, key=lambda p: p.stat().st_size)
    print(f"largest {largest.stat().st_size / 1e6:.2f} MB ({largest.name}) — under the cap")
    _report_version_drift(wheels)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not download; only re-verify the wheels already in vendor/",
    )
    args = parser.parse_args()

    if not args.check:
        if VENDOR.exists():
            shutil.rmtree(VENDOR)
        VENDOR.mkdir()
        _download(VENDOR)
        for wheel in sorted(VENDOR.glob("*.whl")):
            if _distribution_name(wheel.name) in BASE_IMAGE_SATISFIES:
                print(f"dropping {wheel.name} — the base image satisfies it")
                wheel.unlink()

    if not VENDOR.exists():
        print(f"{VENDOR} does not exist; run without --check to build it", file=sys.stderr)
        return 1
    wheels = sorted(VENDOR.glob("*.whl"))
    if not wheels:
        print(f"{VENDOR} holds no wheels", file=sys.stderr)
        return 1
    return _report(wheels)


if __name__ == "__main__":
    sys.exit(main())
