"""Which packs are installed on this checkout — discovered, never named.

The pack editor's tests need two packs: `mock_domain`, the engine's own fixture, which is small and
anchor-free; and a large one, because several guarantees are only meaningful over a file that is big,
comment-heavy and full of YAML anchors. `mock_domain`'s catalog has zero `&` definitions, so the
"nothing re-serialises the file" assertion is vacuous over it and a `safe_load`→`dump` regression
would land green.

Naming that second pack is not an option. A pack is optional by design — a checkout may carry one,
several, or none but the fixture — so a test file that spells a pack name is a test file that only
runs where that pack happens to be installed, leaving the ~10k-line editor uncovered everywhere
else.

So it is chosen by MEASUREMENT: whichever installed pack is biggest by file count, excluding the
fixture. A checkout with only `mock_domain` gets `None` and the tests needing scale skip, and
`test_pack_store.py` carries a fixture-based twin of the anchor guarantee so that skip does not
silence it.

It is not "the configured pack" either: reading `main_config.yaml` would make the suite's strictness
depend on a gitignored file.
"""

from src.utils.paths import REPO_ROOT

PACKS_ROOT = REPO_ROOT / "knowledge"

#: The pack the engine ships as its own neutral fixture. Always present.
FIXTURE_PACK = "mock_domain"


def installed_packs():
    """Every pack directory under `knowledge/`, sorted, by name.

    A directory is a pack if it declares a source catalog — `knowledge/README.md` and any
    stray tooling directory are not packs, and `pack_store` would refuse them anyway.
    """
    if not PACKS_ROOT.is_dir():
        return []
    return sorted(
        p.name
        for p in PACKS_ROOT.iterdir()
        if p.is_dir() and (p / "source_catalog.yaml").is_file()
    )


def scale_pack():
    """The largest installed pack other than the fixture one, or `None`.

    Size is file count rather than bytes: what the tests below need is a pack with many
    files, deep `use_cases/` nesting and anchors, and one 456 KB generated schema would
    win on bytes while giving none of that.
    """
    others = [n for n in installed_packs() if n != FIXTURE_PACK]
    if not others:
        return None
    return max(others, key=lambda n: sum(1 for _ in (PACKS_ROOT / n).rglob("*")))


#: Resolved once at import: the tests use it as a value, and a per-test rediscovery would
#: let a test that writes into a copied tree change what a later test selects.
SCALE_PACK = scale_pack()
