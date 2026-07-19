"""Seed synthetic continuity sessions through real Weft MCP tool functions."""

from __future__ import annotations

from dataclasses import dataclass, field

import asyncpg

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.continuity_manifest import (
    PAAH_CONTINUITY_OTHER_PROJECT_ID,
    PAAH_CONTINUITY_OTHER_USER_ID,
    PAAH_CONTINUITY_PROJECT_ID,
    PAAH_CONTINUITY_USER_ID,
    TURNS,
)
from weft.auth import current_user_id


@dataclass(slots=True)
class SeedContinuityResult:
    episode_id: str
    turn_ids: dict[str, str] = field(default_factory=dict)
    other_project_turn_id: str | None = None
    other_user_turn_id: str | None = None

    @property
    def clean(self) -> bool:
        return len(set(self.turn_ids.values())) == len(TURNS)


async def _seed_episode(ctx, *, user_id: str, project_id: str, title: str):
    from weft.mcp.tools import weft_episode_create

    token = current_user_id.set(user_id)
    try:
        return await weft_episode_create(ctx, title=title, project_id=project_id)
    finally:
        current_user_id.reset(token)


async def _append(ctx, *, user_id: str, episode_id: str, content: str, occurred_at):
    from weft.mcp.tools import weft_turn_append

    token = current_user_id.set(user_id)
    try:
        return await weft_turn_append(
            ctx,
            episode_id=episode_id,
            role="user",
            content=content,
            occurred_at=occurred_at.isoformat(),
        )
    finally:
        current_user_id.reset(token)


async def seed_continuity(pool: asyncpg.Pool) -> SeedContinuityResult:
    app = await build_app_context(pool)
    ctx = make_ctx(app)
    episode = await _seed_episode(
        ctx,
        user_id=PAAH_CONTINUITY_USER_ID,
        project_id=PAAH_CONTINUITY_PROJECT_ID,
        title="Continuity acceptance session",
    )
    result = SeedContinuityResult(episode_id=episode["id"])
    for spec in TURNS:
        turn = await _append(
            ctx,
            user_id=PAAH_CONTINUITY_USER_ID,
            episode_id=result.episode_id,
            content=spec.content,
            occurred_at=spec.occurred_at,
        )
        result.turn_ids[spec.key] = turn["turn_id"]

    other_project = await _seed_episode(
        ctx,
        user_id=PAAH_CONTINUITY_USER_ID,
        project_id=PAAH_CONTINUITY_OTHER_PROJECT_ID,
        title="Other project continuity distractor",
    )
    turn = await _append(
        ctx,
        user_id=PAAH_CONTINUITY_USER_ID,
        episode_id=other_project["id"],
        content="We rejected red in the other project because the mascot disliked it.",
        occurred_at=TURNS[-1].occurred_at,
    )
    result.other_project_turn_id = turn["turn_id"]

    # The project-wide test pool resets every acquired connection to the
    # default test user. Use a dedicated pool for the cross-user distractor so
    # row ownership is authored through the same session-GUC contract a real
    # application connection uses, rather than relying on a superuser fixture.
    async with pool.acquire() as conn:
        host, port = conn._addr
        database = conn._params.database
        username = conn._params.user
        password = conn._params.password

    async def setup_other_user(conn):
        await conn.execute(
            "SELECT set_config('app.user_id', $1, false)",
            PAAH_CONTINUITY_OTHER_USER_ID,
        )

    other_user_pool = await asyncpg.create_pool(
        f"postgresql://{username}:{password}@{host}:{port}/{database}",
        min_size=2,
        max_size=5,
        setup=setup_other_user,
    )
    try:
        other_app = await build_app_context(other_user_pool)
        other_ctx = make_ctx(other_app)
        other_user = await _seed_episode(
            other_ctx,
            user_id=PAAH_CONTINUITY_OTHER_USER_ID,
            project_id=PAAH_CONTINUITY_PROJECT_ID,
            title="Other user continuity distractor",
        )
        turn = await _append(
            other_ctx,
            user_id=PAAH_CONTINUITY_OTHER_USER_ID,
            episode_id=other_user["id"],
            content=(
                "We rejected red because the confidential other-user budget failed."
            ),
            occurred_at=TURNS[-1].occurred_at,
        )
        result.other_user_turn_id = turn["turn_id"]
    finally:
        await other_user_pool.close()
    return result
