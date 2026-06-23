"""Migration 55: add 'failed' to the replay_queue status CHECK constraint.

The aggregation replay executor (E2.L7, loom-ebef8ec1) must drive every claimed
replay_queue row to a TERMINAL status — otherwise the L4 retention guard (which
retains turns for any row WHERE status='pending') pins those turns against
graduation deletion forever, an unbounded retention leak (see weft-99cac4e5).

'done' is the success terminal; this migration adds 'failed' as the
unrecoverable-error terminal so the executor can release a poison row's turns
instead of leaving it pending forever. 'failed' is deliberately NOT retained by
the retention guard (which keys only on 'pending') — a failed replay forfeits
its turns to retention rather than leaking; the re-ask loop re-enqueues a fresh
row if recall still misses.

v53 created the constraint inline as ``status TEXT ... CHECK (status IN
('pending', 'done'))``, which Postgres names ``replay_queue_status_check``.

Spec: Loom task loom-ebef8ec1 (E2.L7).
"""

from __future__ import annotations

VERSION = 55
DESCRIPTION = "replay_queue: add 'failed' terminal status to the status CHECK"
SQL = r"""
ALTER TABLE replay_queue DROP CONSTRAINT IF EXISTS replay_queue_status_check;
ALTER TABLE replay_queue ADD CONSTRAINT replay_queue_status_check
    CHECK (status IN ('pending', 'done', 'failed'));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
