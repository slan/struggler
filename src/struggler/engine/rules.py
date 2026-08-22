from __future__ import annotations

from struggler.engine.data_loader import load_json

RULES: dict = load_json("rules.json")

#: Bumped whenever a change alters how the rules resolve -- a fix, a
#: clarified ruling, a data correction -- so that anything measured on the
#: engine (a bot's rating, a trained policy) can say which game it was
#: measured on. 1 is the engine before the August 2026 fixes (a scoring
#: card could be held past the turn, Military Ops uncapped, ...).
RULES_VERSION: int = int(RULES["rules_version"])
