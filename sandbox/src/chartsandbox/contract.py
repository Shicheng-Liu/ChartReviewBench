"""Task-instance contract — the interface between the data pipeline and the sandbox.

The data team produces task instances on disk; the sandbox only ever reads this
schema. Keeping this contract stable lets both sides develop independently.

Layout of one task instance:

    tasks/<task_id>/
    ├── task.yaml        # this schema
    ├── workspace/       # files copied into the agent's writable workspace
    └── oracle/          # reference files; read-only, NEVER exposed to the agent

Core abstraction:  a task is an ordered list of subgoals; each subgoal carries a
set of verifiers. This unifies all task families (reproduce_edit / data_insight /
debug_repair / multi_chart) and yields long-horizon progress curves for free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

TaskFamily = Literal["reproduce_edit", "data_insight", "debug_repair", "multi_chart"]


class VerifierSpec(BaseModel):
    """One check attached to a subgoal. `type` maps to a registered verifier."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class Subgoal(BaseModel):
    id: str
    desc: str = ""
    weight: float = 1.0
    verifiers: list[VerifierSpec] = Field(default_factory=list)


class Task(BaseModel):
    id: str
    family: TaskFamily
    instruction: str
    horizon_hint: int | None = None
    workspace_files: list[str] = Field(default_factory=list)
    subgoals: list[Subgoal]

    # Runtime budget (sandbox-enforced). `None` means *uncapped*: the released
    # A/B/C suite selects this deliberately (protocol `abc-uncapped-v1`), so that an
    # episode ends on the agent's own judgement rather than on a harness ceiling.
    # The finite guards that remain are per-step: `step_timeout_s`, and the HTTP and
    # kernel timeouts underneath it.
    step_budget: int | None = 30
    # Iteration budget for the tagged-text protocol: one iteration is one
    # reason -> code -> execute -> inspect cycle. The tool-calling protocol ignores
    # it and spends `step_budget` directly, one tool call per step.
    max_turns: int | None = 10
    step_timeout_s: int = 30
    wall_time_s: int | None = 900

    @model_validator(mode="after")
    def _check(self) -> "Task":
        if not self.subgoals:
            raise ValueError(f"task {self.id!r} has no subgoals")
        ids = [s.id for s in self.subgoals]
        if len(ids) != len(set(ids)):
            raise ValueError(f"task {self.id!r} has duplicate subgoal ids")
        for field in ("step_budget", "max_turns", "wall_time_s"):
            value = getattr(self, field)
            if value is not None and value < 1:
                raise ValueError(f"task {self.id!r} has {field}={value}; use a positive "
                                 f"value, or null for uncapped")
        # Deliberately no cross-check between step_budget and max_turns. An
        # interaction-budget sweep runs the same instance at 1, 2, 3, 5, 10 steps, and
        # a task is not malformed for having fewer steps than the tagged-text default
        # would like — the agent reconciles the two at reset and reports having done
        # so (see TaggedAgent.max_turns_clamped), rather than the task failing to load.
        return self


class LoadedTask(BaseModel):
    """A validated task plus resolved on-disk locations."""

    model_config = {"arbitrary_types_allowed": True}

    task: Task
    task_dir: Path
    workspace_dir: Path  # task_dir/workspace  (template; copied per run)
    oracle_dir: Path     # task_dir/oracle      (read-only reference)


def load_task(task_dir: str | Path) -> LoadedTask:
    task_dir = Path(task_dir).resolve()
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"no task.yaml in {task_dir}")
    data = yaml.safe_load(yaml_path.read_text())
    task = Task(**data)

    ws = task_dir / "workspace"
    for f in task.workspace_files:
        if not (ws / f).exists():
            raise FileNotFoundError(
                f"task {task.id!r} declares workspace file {f!r} but {ws / f} is missing"
            )
    return LoadedTask(
        task=task,
        task_dir=task_dir,
        workspace_dir=ws,
        oracle_dir=task_dir / "oracle",
    )
