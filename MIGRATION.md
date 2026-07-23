# Migration Guide: v0.6.x to v0.7.0 (AI SDK 7)

Agent v0.7 targets AI SDK 7. Upgrade the Agent package, AI SDK core, provider
utilities, and provider packages together:

```bash
pnpm add @convex-dev/agent@^0.7.0 ai@^7.0.0 \
  @ai-sdk/provider-utils@^5.0.0 \
  @ai-sdk/openai@^4.0.0
```

Use the corresponding v4 release for other official `@ai-sdk/*` providers.
Third-party providers must explicitly support AI SDK 7.

## Required API changes

### Agent prompts

Rename `system` to `instructions`:

```ts
await agent.generateText(ctx, { threadId }, {
  instructions: "You are a helpful assistant.",
  prompt: "Hello",
});
```

### Step control and callbacks

AI SDK 7 renamed `stepCountIs` to `isStepCount` and `onStepFinish` to
`onStepEnd`:

```ts
import { isStepCount } from "ai";

await agent.generateText(ctx, { threadId }, {
  prompt: "Research this topic",
  stopWhen: isStepCount(5),
  onStepEnd: async (step) => {
    console.log(step.finishReason);
  },
});
```

### Tools

`createTool` now follows AI SDK 7's `inputSchema` and `execute` names. The
Agent runtime context is the first argument, followed by validated input and
the AI SDK execution options:

```ts
const search = createTool({
  description: "Search documents",
  inputSchema: z.object({ query: z.string() }),
  execute: async (ctx, input, options) => {
    return await ctx.runQuery(api.documents.search, {
      query: input.query,
    });
  },
});
```

If you add custom runtime fields, pass them on the generation context and type
the tool context as before.

### Usage and raw responses

Usage now follows AI SDK 7's input/output naming:

- `inputTokens` replaces prompt-token fields.
- `outputTokens` replaces completion-token fields.
- Cache and reasoning counts live under `inputTokenDetails` and
  `outputTokenDetails`.

Raw response bodies are not collected by default. Enable the AI SDK raw-body
option only where a handler genuinely needs them; doing so can materially
increase memory use.

### Model IDs

Both AI SDK 7 language-model objects and registry IDs such as
`"openai:gpt-4o-mini"` are supported.

## Persisted messages and streams

No data migration is required. New streams are written with the
`UIMessageChunkV7` marker. Streams written by Agent v0.6 retain their
`UIMessageChunk` marker and are replayed with the pinned AI SDK 6 semantics, so
aborted or in-flight streams remain recoverable after deployment.

The persisted message format is additive: AI SDK 7 custom parts,
reasoning-file parts, and canonical tool-result file parts are retained.
Provider references and inline text file data round-trip without being
flattened. Large binary file data uses Agent's tracked Convex file storage and
participates in the existing reference-count cleanup lifecycle.

## Verification

After updating call sites and reinstalling dependencies, run:

```bash
vp run typecheck
vp test
vp run lint
```

For the upstream SDK changes, see the
[AI SDK 7 migration guide](https://ai-sdk.dev/docs/migration-guides/migration-guide-7-0).
