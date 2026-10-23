# MIDIBench: agentic evals for symbolic music editing

A [Pi](https://github.com/earendil-works/pi-mono) coding-agent extension that gives an agent MIDI tools, plus a tiered eval suite with deterministic music-theory graders. It is built to answer one question: **does the representation an agent sees, and its access to self-verification, change how reliably it completes multi-step music edits?**

Status: weeks 1-2 of the build plan. midicore, the three views, edit ops, the shared checker, 10 T1 tasks with gold solutions, the Pi extension and the harness all run end to end. T2-T4 tasks, the harmony/voice-leading graders and the LLM judge come next.

## Quickstart

```bash
uv sync                                   # Python 3.12+, installs midicore/graders/harness
uv run pytest                             # unit tests, view round trips, task sanity checks
uv run midibench check-tasks              # every task fails on its input and passes on its gold
uv run python -m midicore read --file tasks/fixtures/chorale_bwv347.mid --view musical --bars 1-4
```

Run the agent (needs `pi` on PATH and credentials for the model's provider):

```bash
uv run midibench run --tasks t1 --split dev --model anthropic/claude-sonnet-5 \
    --views raw,notes,musical --verify 0,1 --trials 5 --run-id my-run
uv run midibench report my-run            # re-print tables; results/my-run/report.md
```

`--model ollama/<id>` runs a local Ollama model without touching `~/.pi/agent/models.json`. Re-running with the same `--run-id` resumes: finished trials are skipped.

## How it fits together

```
harness (Python)  ──subprocess──▶  pi -p --mode json  + pi-ext/index.ts  ──subprocess──▶  python -m midicore
 tasks x configs x trials           midi_read / midi_analyze /                             views, analysis,
 sandbox per trial, grading,        midi_edit / midi_verify, JSONL log,                    ops, checks
 results.jsonl + .parquet           tool-call budget, sandbox path guard
```

The graders and the agent's tools share one library (`midicore`), so they cannot disagree about what a bar, a key or a correct edit is. `midi_verify` runs the very checks the graders use; whether having it helps is RQ2.

| Path | What it is |
|---|---|
| `midicore/model.py` | `Note`, `Score`, `BarMap`. Beats are quarter notes; bars are 1-based, a pickup is bar 0 |
| `midicore/io.py` | `.mid` load/save (the only tick-aware code) |
| `midicore/views/` | `raw`, `notes`, `musical` renderers, each with a `parse()` used to prove it is lossless |
| `midicore/analysis.py` | key finding (Aarden-Essen profiles), per-beat chord labels, roman numerals |
| `midicore/ops.py` | typed edit ops (Pydantic), atomic apply, diff summary |
| `midicore/verify.py` | constraint checks shared by `midi_verify` and the graders |
| `pi-ext/` | the Pi extension (`index.ts`) and a local Ollama provider (`local-models.ts`) |
| `tasks/t1/` | task YAMLs; `tasks/fixtures/` holds inputs and `build_fixtures.py` |
| `graders/registry.py` | grader lookup and task-level pass/partial scoring |
| `harness/` | `agent.py` (Pi invocation), `run.py` (grid, resume), `metrics.py`, `report.py`, `cli.py` |

## The views (RQ1's independent variable)

All three carry the same notes, tempo map, meter map, key signatures, track names, channels and programs; tests assert `parse(render(score)) == score` for each view on every fixture.

- **raw**: the MIDI event stream with delta ticks (`dt=480 note_off ch=0 n=72 v=0`). Bar math is left to the agent.
- **notes**: one line per note in musical time (`[Soprano] bar 3 beat 2  D5  dur 0.5  vel 80`). Note names follow the declared key signature's sharps/flats.
- **musical**: notes grouped by bar and track (`Soprano v80: C5(1) D5(0.5) r(0.5) F5(2)`), plus the estimated key, a chord label per beat (`ii7 (Cm7)`) and scale degrees. The analysis is automatic and can be wrong; over-trusting it is a failure mode we expect to see.

## Adding a task

1. Put the input in `tasks/fixtures/` (or generate it in `build_fixtures.py`).
2. Write `tasks/tN/<id>.yaml` (the file name must equal `id`):

```yaml
id: t1-transpose-002
tier: 1
category: transpose
input: fixtures/leadsheet_c.mid
instruction: >
  Transpose the Melody up a perfect fourth in bars 5-8 only.
graders:
  - {name: transpose_exact, args: {semitones: 5, tracks: [Melody], bars: [5, 8]}}
  - {name: preserve, args: {tracks: [Melody], bars: [[1, 4], [9, null]]}}
  - {name: preserve, args: {exclude_tracks: [Melody]}}
weights: {transpose_exact: 2}       # partial-credit weights, optional
pass_rule: all                      # all | threshold:0.8
split: dev                          # dev | test (held out: never iterate on it)
gold_ops:                           # or `gold: fixtures/clean.mid`
  - {op: transpose, semitones: 5, scope: {tracks: [Melody], bars: [5, 8]}}
```

3. `uv run midibench check-tasks` must report it ok, and it is worth adding a near-miss answer to `NEAR_MISSES` in `tests/test_tasks_harness.py` to prove the graders reject it.

Graders available now: `preserve`, `transpose_exact`, `range`, `tempo`, `quantize`, `key`, `empty`, `velocity`, `contains`, `preserve_pitches` (see `midicore/verify.py`).

## Evaluating a new model

Any model Pi can reach works: `--model provider/id`. Each trial gets a sandbox under `results/<run_id>/trials/<task>/<config>/<trial>/` with `input.mid` (read-only), the edited `work.mid`, Pi's event stream (`pi_events.jsonl`) and the extension's tool log (`tools.jsonl`). One row per trial goes to `results/<run_id>/results.jsonl` and is mirrored to `results.parquet`. `results/` is gitignored.

## Notes from the Pi spike (pi 0.84.1)

- Headless: `pi -p --mode json --no-session`, with stdin closed. With an open stdin pipe, `-p` waits for input forever.
- Pi exits 0 even when the provider call fails (expired key, 401). The harness reads the final assistant message's `stopReason` to tell an agent error from a real attempt.
- `--no-builtin-tools` removes bash/read/edit/write, so the agent only has `midi_*` tools; `--no-extensions --no-skills --no-context-files --no-prompt-templates --no-approve` keep runs hermetic.
- Pi has no max-turns flag. The extension counts calls at `tool_execution_start` and blocks with `terminate: true` past the budget.
- Schema-validation failures never reach `tool_result`; the extension logs from `tool_execution_end`, which sees them.
- Small models often send nested JSON as a string. `prepareArguments` accepts that (identically in every condition) and logs a `coerced_args` event.
