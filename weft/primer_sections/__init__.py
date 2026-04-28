"""Primer section builders — modular components of build_primer.

Each budget-packing section exposes three entry points:
  * ``fetch_<name>_section(ctx)``  — async, pure I/O, returns SectionFetch
  * ``pack_<name>_section(ctx, f)`` — sync, mutates ctx, returns SectionResult
  * ``build_<name>_section(ctx)``  — thin wrapper: fetch then pack

The orchestrator runs all fetches in parallel (one concurrent DB batch),
then packs sequentially in priority order so shared ctx state
(used_tokens, seen_ids, ...) is updated correctly. Tests and other
callers can continue using ``build_<name>_section`` as a single-shot entry.
"""

from weft.primer_sections.anti_patterns import (
    build_anti_patterns_section,
    fetch_anti_patterns_section,
    pack_anti_patterns_section,
)
from weft.primer_sections.autonomy import (
    build_autonomy_section,
    fetch_autonomy_section,
    pack_autonomy_section,
)
from weft.primer_sections.behaviors import (
    build_behaviors_section,
    fetch_behaviors_section,
    pack_behaviors_section,
)
from weft.primer_sections.calibration import (
    build_calibration_section,
    fetch_calibration_section,
    pack_calibration_section,
)
from weft.primer_sections.changes_since import build_changes_since_section
from weft.primer_sections.context import PrimerContext, SectionFetch, SectionResult
from weft.primer_sections.cost import (
    build_cost_section,
    fetch_cost_section,
    pack_cost_section,
)
from weft.primer_sections.decisions import (
    build_decisions_section,
    fetch_decisions_section,
    pack_decisions_section,
)
from weft.primer_sections.degradation import (
    build_degradation_section,
    fetch_degradation_section,
    pack_degradation_section,
)
from weft.primer_sections.disclosure import apply_progressive_disclosure
from weft.primer_sections.entities import (
    build_entities_section,
    fetch_entities_section,
    pack_entities_section,
)
from weft.primer_sections.grounding import (
    build_grounding_section,
    fetch_grounding_section,
    pack_grounding_section,
)
from weft.primer_sections.handoff import (
    build_handoff_section,
    fetch_handoff_section,
    pack_handoff_section,
)
from weft.primer_sections.issues import (
    build_issues_section,
    fetch_issues_section,
    pack_issues_section,
)
from weft.primer_sections.onboarding import build_onboarding_section
from weft.primer_sections.recent_work import (
    build_recent_work_section,
    fetch_recent_work_section,
    pack_recent_work_section,
)
from weft.primer_sections.rules import (
    build_rules_section,
    fetch_rules_section,
    pack_rules_section,
)
from weft.primer_sections.triggers import (
    build_triggers_section,
    fetch_triggers_section,
    pack_triggers_section,
)
from weft.primer_sections.wellness import build_wellness_section
from weft.primer_sections.working_memory import (
    build_working_memory_section,
    fetch_working_memory_section,
    pack_working_memory_section,
)

__all__ = [
    "PrimerContext",
    "SectionFetch",
    "SectionResult",
    "apply_progressive_disclosure",
    "build_anti_patterns_section",
    "build_autonomy_section",
    "build_behaviors_section",
    "build_calibration_section",
    "build_changes_since_section",
    "build_cost_section",
    "build_decisions_section",
    "build_degradation_section",
    "build_entities_section",
    "build_grounding_section",
    "build_handoff_section",
    "build_issues_section",
    "build_onboarding_section",
    "build_recent_work_section",
    "build_rules_section",
    "build_triggers_section",
    "build_wellness_section",
    "build_working_memory_section",
    "fetch_anti_patterns_section",
    "fetch_autonomy_section",
    "fetch_behaviors_section",
    "fetch_calibration_section",
    "fetch_cost_section",
    "fetch_decisions_section",
    "fetch_degradation_section",
    "fetch_entities_section",
    "fetch_grounding_section",
    "fetch_handoff_section",
    "fetch_issues_section",
    "fetch_recent_work_section",
    "fetch_rules_section",
    "fetch_triggers_section",
    "fetch_working_memory_section",
    "pack_anti_patterns_section",
    "pack_autonomy_section",
    "pack_behaviors_section",
    "pack_calibration_section",
    "pack_cost_section",
    "pack_decisions_section",
    "pack_degradation_section",
    "pack_entities_section",
    "pack_grounding_section",
    "pack_handoff_section",
    "pack_issues_section",
    "pack_recent_work_section",
    "pack_rules_section",
    "pack_triggers_section",
    "pack_working_memory_section",
]
