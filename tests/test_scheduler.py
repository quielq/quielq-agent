import json

from quielq_agent import scheduler
from quielq_agent.config import AgentConfig, ScheduleEntry
from quielq_agent.tools import ToolRegistry


def _config(tmp_path, **overrides):
    # memory_dir as an absolute path: Path(PROJECT_ROOT) / absolute_path resolves
    # to just the absolute path, so this points memory_path at tmp_path without
    # needing to touch the real project's fleet/memory/ directory.
    defaults = dict(
        name="ops-watch",
        display_name="Ops Watch",
        model="m",
        system_prompt_file="fleet/prompts/ops-watch.md",
        memory_dir=str(tmp_path / "mem"),
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


async def test_run_scheduled_entry_writes_experiences_jsonl(monkeypatch, tmp_path):
    async def fake_run_turn(config, messages, registry):
        return messages + [{"role": "assistant", "content": "all clear"}]

    monkeypatch.setattr(scheduler, "run_turn", fake_run_turn)
    monkeypatch.setattr(AgentConfig, "load_system_prompt", lambda self: "you are ops-watch")

    config = _config(tmp_path)
    entry = ScheduleEntry(cron="0 */6 * * *", message="check things", deliver_to="memory")

    record = await scheduler.run_scheduled_entry(config, entry, ToolRegistry())

    assert record["status"] == "ok"
    assert record["result"] == "all clear"

    files = list((tmp_path / "mem" / "experiences").glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["agent"] == "ops-watch"
    assert saved["trigger"] == "check things"


async def test_run_scheduled_entry_notes_unimplemented_delivery(monkeypatch, tmp_path):
    async def fake_run_turn(config, messages, registry):
        return messages + [{"role": "assistant", "content": "done"}]

    monkeypatch.setattr(scheduler, "run_turn", fake_run_turn)
    monkeypatch.setattr(AgentConfig, "load_system_prompt", lambda self: "prompt")

    config = _config(tmp_path)
    entry = ScheduleEntry(cron="0 6 * * *", message="hi", deliver_to="telegram")

    record = await scheduler.run_scheduled_entry(config, entry, ToolRegistry())

    assert "isn't implemented yet" in record["note"]


async def test_run_scheduled_entry_records_errors(monkeypatch, tmp_path):
    from quielq_agent.llm import OllamaCallError

    async def failing_run_turn(config, messages, registry):
        raise OllamaCallError("network blip")

    monkeypatch.setattr(scheduler, "run_turn", failing_run_turn)
    monkeypatch.setattr(AgentConfig, "load_system_prompt", lambda self: "prompt")

    config = _config(tmp_path)
    entry = ScheduleEntry(cron="0 6 * * *", message="hi", deliver_to="memory")

    record = await scheduler.run_scheduled_entry(config, entry, ToolRegistry())

    assert record["status"] == "error"
    assert "network blip" in record["result"]


async def test_scheduler_loop_once_runs_every_entry(monkeypatch, tmp_path):
    calls = []

    async def fake_run_scheduled_entry(config, entry, registry):
        calls.append(entry.message)
        return {}

    monkeypatch.setattr(scheduler, "run_scheduled_entry", fake_run_scheduled_entry)

    config = _config(
        tmp_path,
        schedule=[
            {"cron": "0 6 * * *", "message": "a"},
            {"cron": "0 18 * * *", "message": "b"},
        ],
    )

    await scheduler._scheduler_loop(config, ToolRegistry(), run_once=True)

    assert calls == ["a", "b"]


async def test_scheduler_loop_no_schedule_is_a_noop(monkeypatch, tmp_path):
    called = False

    async def fake_run_scheduled_entry(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(scheduler, "run_scheduled_entry", fake_run_scheduled_entry)

    config = _config(tmp_path, name="echo", display_name="Echo")
    await scheduler._scheduler_loop(config, ToolRegistry(), run_once=True)

    assert not called
