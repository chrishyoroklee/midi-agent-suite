// Registers a local Ollama provider so `--model ollama/<id>` works without
// editing ~/.pi/agent/models.json. The harness loads this only for ollama/* models.
//   MIDIBENCH_OLLAMA_MODEL   model id to register, e.g. llama3.1:8b
//   MIDIBENCH_OLLAMA_URL     default http://localhost:11434/v1

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const id = process.env.MIDIBENCH_OLLAMA_MODEL;
  if (!id) return;
  pi.registerProvider("ollama", {
    baseUrl: process.env.MIDIBENCH_OLLAMA_URL ?? "http://localhost:11434/v1",
    api: "openai-completions",
    apiKey: "ollama",
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
    models: [
      {
        id,
        name: id,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32768,
        maxTokens: 4096,
      },
    ],
  } as any);
}
