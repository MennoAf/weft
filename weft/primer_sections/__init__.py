"""Primer section builders — modular components of build_primer.

Re-exports PrimerContext, SectionResult, and all section builder functions
so callers can import from the package directly:

    from weft.primer_sections import PrimerContext, build_rules_section
"""

from weft.primer_sections.anti_patterns import build_anti_patterns_section
from weft.primer_sections.behaviors import build_behaviors_section
from weft.primer_sections.changes_since import build_changes_since_section
from weft.primer_sections.context import PrimerContext, SectionResult
from weft.primer_sections.decisions import build_decisions_section
from weft.primer_sections.disclosure import apply_progressive_disclosure
from weft.primer_sections.entities import build_entities_section
from weft.primer_sections.grounding import build_grounding_section
from weft.primer_sections.handoff import build_handoff_section
from weft.primer_sections.issues import build_issues_section
from weft.primer_sections.onboarding import build_onboarding_section
from weft.primer_sections.recent_work import build_recent_work_section
from weft.primer_sections.rules import build_rules_section
from weft.primer_sections.wellness import build_wellness_section
from weft.primer_sections.working_memory import build_working_memory_section

__all__ = [
    "PrimerContext",
    "SectionResult",
    "apply_progressive_disclosure",
    "build_anti_patterns_section",
    "build_behaviors_section",
    "build_changes_since_section",
    "build_decisions_section",
    "build_entities_section",
    "build_grounding_section",
    "build_handoff_section",
    "build_issues_section",
    "build_onboarding_section",
    "build_recent_work_section",
    "build_rules_section",
    "build_wellness_section",
    "build_working_memory_section",
]
