// MIDIBench Pi extension: midi_* tools backed by `python -m midicore`.
//
// Deliberately thin: tool schemas, a subprocess call, a sandbox path guard and
// trajectory logging. All music logic lives in midicore (Python).
//
// Environment (set per trial by the harness):
//   MIDIBENCH_VIEW             raw | notes | musical   view returned by midi_read (default notes)
//   MIDIBENCH_VERIFY           1 registers midi_verify (default 0)
//   MIDIBENCH_LOG              JSONL file for the trajectory log
//   MIDIBENCH_PYTHON           python executable that can import midicore (default python3)
//   MIDIBENCH_MAX_TOOL_CALLS   tool-call budget per run (default 30)
//   MIDIBENCH_PROTECT          comma-separated file names midi_edit refuses (default input.mid)
//   MIDIBENCH_REFERENCE        original file for midi_verify comparisons (default input.mid)

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { appendFileSync } from "node:fs";
import { relative, resolve, isAbsolute } from "node:path";

const VIEW = process.env.MIDIBENCH_VIEW ?? "notes";
const VERIFY = process.env.MIDIBENCH_VERIFY === "1";
const LOG = process.env.MIDIBENCH_LOG;
const PYTHON = process.env.MIDIBENCH_PYTHON ?? "python3";
const MAX_CALLS = Number(process.env.MIDIBENCH_MAX_TOOL_CALLS ?? "30");
const PROTECT = process.env.MIDIBENCH_PROTECT ?? "input.mid";
const REFERENCE = process.env.MIDIBENCH_REFERENCE ?? "input.mid";

const OPS_DOC = `Apply a list of edit operations to a MIDI file, atomically: if any op fails, nothing is written and an error names the failing op.
Positions use bars (1-based; a pickup bar is bar 0) and beats within the bar (1-based, in quarter notes: 1, 1.5, 2, ...). Pitches are names like "C4" (middle C), "Bb3", "F#5" or MIDI numbers.
A selector/scope picks notes by onset; every given field must match, omitted fields match everything:
  {"track": "Bass"} {"tracks": ["S","A"]} {"bars": [5, 8]} (inclusive) {"bar": 3, "beat": 2.5} {"pitch": "C4"} {"pitches": ["C4","E4"]}
Ops:
  {"op":"transpose","semitones":-2,"scope":{...}}
  {"op":"quantize","grid":0.25,"strength":1.0,"ends":false,"scope":{...}}   grid in beats: 0.5=8th, 0.25=16th, 0.3333333=8th triplet
  {"op":"insert_note","track":"Bass","bar":5,"beat":1,"pitch":"C3","dur":2,"vel":80}
  {"op":"delete_notes","selector":{...}}
  {"op":"move_note","selector":{...},"dpitch":1,"dbeat":0}
  {"op":"set_velocity","selector":{...},"value":64}
  {"op":"set_duration","selector":{...},"dur":0.5}
  {"op":"set_tempo","bpm":120,"at_bar":1,"at_beat":1}
  {"op":"set_time_signature","num":3,"den":4,"at_bar":1}
  {"op":"replace_bars","track":"Bass","bars":[5,8],"notes":[{"bar":5,"beat":1,"pitch":"C3","dur":2,"vel":80}, ...]}   replaces every note of the track in those bars
  {"op":"add_track","name":"Alto","channel":2,"program":52}
  {"op":"remove_track","name":"Alto"}
Returns a diff summary: notes added/removed per track and the bars touched.`;

const VERIFY_DOC = `Check constraints against a MIDI file and get pass/fail plus details for each.
Each constraint is {"name": ..., "args": {...}}. Most checks accept scope args: "tracks" (list or "all"), "track", "exclude_tracks", "bars" ([first, last], or [[a,b],[c,d]]; last may be null for "to the end").
Checks that compare against the original compare with ${REFERENCE}.
  preserve          notes in scope are identical to the original (pitch, start, dur); args: velocity (bool)
  transpose_exact   notes in scope equal the original shifted by N semitones; args: semitones, allow_octave
  range             all notes in scope within [low, high]; args: low, high (e.g. "E2", "C4")
  tempo             tempo throughout the scope's bars equals bpm; args: bpm, bars
  quantize          every onset in scope is on the grid; args: grid (beats), tolerance (default 0.01)
  key               estimated key of the scope is the given key; args: key ("D major", "G minor"), diatonic (bool)
  empty             there are no notes in scope
  velocity          every note in scope has velocity value; args: value, tolerance
  contains          a note exists; args: track, bar, beat, pitch, dur (optional)
  preserve_pitches  same pitches and note count as the original in scope; timing may change`;

type CoreResult = { ok: boolean; result?: any; error?: any };

function logEvent(event: Record<string, unknown>) {
  if (!LOG) return;
  appendFileSync(LOG, JSON.stringify({ ts: Date.now(), ...event }) + "\n");
}

// Small models often send nested JSON as a string ("[{...}]") or a single object
// instead of an array. Accept both (identically in every condition) and log it,
// so tool-format quirks do not masquerade as musical failures.
function coerceList(args: unknown, key: string, tool: string): any {
  if (!args || typeof args !== "object") return args;
  const input = { ...(args as Record<string, unknown>) };
  let value = input[key];
  const original = value;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return args;
    }
  }
  if (value && typeof value === "object" && !Array.isArray(value)) value = [value];
  if (value !== original) {
    input[key] = value;
    logEvent({ type: "coerced_args", tool, key, from: typeof original });
  }
  return input;
}

function sandboxPath(cwd: string, file: string): string {
  const cleaned = file.replace(/^@/, "");
  const abs = resolve(cwd, cleaned);
  const rel = relative(cwd, abs);
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw new Error(`Access denied: ${file} is outside the working directory. Use files in the current directory, e.g. work.mid.`);
  }
  return abs;
}

export default function (pi: ExtensionAPI) {
  async function runCore(args: string[], signal?: AbortSignal, cwd?: string): Promise<CoreResult> {
    const r = await pi.exec(PYTHON, ["-m", "midicore", ...args], { signal, timeout: 60_000, cwd });
    try {
      return JSON.parse(r.stdout.trim().split("\n").pop() ?? "") as CoreResult;
    } catch {
      throw new Error(`midicore failed (exit ${r.code}): ${(r.stderr || r.stdout).slice(-2000)}`);
    }
  }

  function toolResult(res: CoreResult, render: (result: any) => string) {
    if (!res.ok) throw new Error(JSON.stringify(res.error));
    return { content: [{ type: "text" as const, text: render(res.result) }], details: {} };
  }

  pi.registerTool({
    name: "midi_read",
    label: "Read MIDI",
    description: "Read a MIDI file as text. Optionally limit to a bar range (e.g. \"5-8\") and/or tracks (comma-separated names).",
    parameters: Type.Object({
      file: Type.String({ description: "path to a .mid file in the working directory" }),
      bars: Type.Optional(Type.String({ description: "bar range like \"5-8\" or \"3\"" })),
      tracks: Type.Optional(Type.String({ description: "comma-separated track names" })),
    }),
    async execute(_id, p, signal, _onUpdate, ctx) {
      const file = sandboxPath(ctx.cwd, p.file);
      const args = ["read", "--file", file, "--view", VIEW];
      if (p.bars) args.push("--bars", p.bars);
      if (p.tracks) args.push("--tracks", p.tracks);
      return toolResult(await runCore(args, signal, ctx.cwd), (r) => r.text);
    },
  });

  pi.registerTool({
    name: "midi_analyze",
    label: "Analyze MIDI",
    description: "Analyze a MIDI file: estimated key (with alternatives), meter, tempo, bar count, per-track range/density/quantization, and a chord label per beat.",
    parameters: Type.Object({ file: Type.String({ description: "path to a .mid file in the working directory" }) }),
    async execute(_id, p, signal, _onUpdate, ctx) {
      const file = sandboxPath(ctx.cwd, p.file);
      return toolResult(await runCore(["analyze", "--file", file], signal, ctx.cwd), (r) => JSON.stringify(r, null, 1));
    },
  });

  pi.registerTool({
    name: "midi_edit",
    label: "Edit MIDI",
    description: OPS_DOC,
    parameters: Type.Object({
      file: Type.String({ description: "path to the .mid file to edit (it is rewritten in place)" }),
      ops: Type.Array(Type.Object({ op: Type.String() }, { additionalProperties: true }), { description: "list of op objects, applied in order" }),
    }),
    prepareArguments: (args) => coerceList(args, "ops", "midi_edit"),
    async execute(_id, p, signal, _onUpdate, ctx) {
      const file = sandboxPath(ctx.cwd, p.file);
      const args = ["edit", "--file", file, "--ops", JSON.stringify(p.ops), "--protect", PROTECT];
      return toolResult(await runCore(args, signal, ctx.cwd), (r) => JSON.stringify(r, null, 1));
    },
  });

  if (VERIFY) {
    pi.registerTool({
      name: "midi_verify",
      label: "Verify MIDI",
      description: VERIFY_DOC,
      parameters: Type.Object({
        file: Type.String({ description: "path to the .mid file to check" }),
        constraints: Type.Array(Type.Object({ name: Type.String(), args: Type.Optional(Type.Object({}, { additionalProperties: true })) }), {
          description: "list of {name, args} constraints",
        }),
      }),
      prepareArguments: (args) => coerceList(args, "constraints", "midi_verify"),
      async execute(_id, p, signal, _onUpdate, ctx) {
        const file = sandboxPath(ctx.cwd, p.file);
        const args = ["verify", "--file", file, "--constraints", JSON.stringify(p.constraints), "--reference", resolve(ctx.cwd, REFERENCE)];
        return toolResult(await runCore(args, signal, ctx.cwd), (r) => JSON.stringify(r, null, 1));
      },
    });
  }

  // Count every call at execution start (including ones that fail schema
  // validation), enforce the budget in tool_call, and log at execution end,
  // which fires for validation failures too.
  let calls = 0;
  const argsById = new Map<string, unknown>();

  pi.on("tool_execution_start", async (event) => {
    calls += 1;
    argsById.set(event.toolCallId, event.args);
  });

  pi.on("tool_call", async (event) => {
    if (calls > MAX_CALLS) {
      logEvent({ type: "budget_exhausted", tool: event.toolName, input: event.input, call: calls });
      return { block: true, reason: `Tool-call budget of ${MAX_CALLS} exhausted; stop now.`, terminate: true };
    }
  });

  pi.on("tool_execution_end", async (event) => {
    const content = (event.result as any)?.content ?? [];
    const text = content.map((c: any) => (c.type === "text" ? c.text : "")).join("\n");
    logEvent({
      type: "tool_result",
      call: calls,
      tool: event.toolName,
      input: argsById.get(event.toolCallId),
      isError: event.isError,
      output: text.slice(0, 8000),
    });
    argsById.delete(event.toolCallId);
  });

  pi.on("session_start", async () => {
    logEvent({ type: "session_start", view: VIEW, verify: VERIFY, max_tool_calls: MAX_CALLS });
  });
}
