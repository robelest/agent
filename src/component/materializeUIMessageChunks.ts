import type {
  Message,
  MessageContentParts,
  MessageWithMetadataInternal,
  ProviderMetadata,
  StreamDelta,
  StreamMessage,
} from "../validators.js";

type RecordValue = Record<string, unknown>;
type AssistantContentPart = Exclude<
  Extract<Message, { role: "assistant" }>["content"],
  string
>[number];
type ToolContentPart = Extract<Message, { role: "tool" }>["content"][number];

type TextPart = {
  type: "text";
  text: string;
  providerMetadata?: ProviderMetadata;
};

type ReasoningPart = {
  type: "reasoning";
  text: string;
  providerMetadata?: ProviderMetadata;
};

type FilePart = {
  type: "file";
  url: string;
  mediaType: string;
  filename?: string;
  providerMetadata?: ProviderMetadata;
};

type ReasoningFilePart = {
  type: "reasoning-file";
  url: string;
  mediaType: string;
  providerMetadata?: ProviderMetadata;
};

type CustomPart = {
  type: "custom";
  kind: string;
  providerMetadata?: ProviderMetadata;
};

type SourcePart =
  | {
      type: "source-url";
      sourceId: string;
      url: string;
      title?: string;
      providerMetadata?: ProviderMetadata;
    }
  | {
      type: "source-document";
      sourceId: string;
      mediaType: string;
      title: string;
      filename?: string;
      providerMetadata?: ProviderMetadata;
    };

type ToolPart = {
  type: `tool-${string}` | "dynamic-tool";
  toolName?: string;
  toolCallId: string;
  state:
    | "input-streaming"
    | "input-available"
    | "approval-requested"
    | "approval-responded"
    | "output-available"
    | "output-error"
    | "output-denied";
  input?: unknown;
  rawInput?: unknown;
  output?: unknown;
  errorText?: string;
  providerExecuted?: boolean;
  callProviderMetadata?: ProviderMetadata;
  resultProviderMetadata?: ProviderMetadata;
  preliminary?: boolean;
  title?: string;
  toolMetadata?: RecordValue;
  approval?: {
    id: string;
    approved?: boolean;
    reason?: string;
    isAutomatic?: boolean;
    signature?: string;
  };
};

type DataPart = {
  type: `data-${string}`;
  id?: string;
  data: unknown;
};

type LegacyUIMessagePart =
  | TextPart
  | ReasoningPart
  | FilePart
  | ReasoningFilePart
  | CustomPart
  | SourcePart
  | ToolPart
  | DataPart
  | { type: "step-start" };

type PartialToolCall = {
  text: string;
  toolName: string;
  dynamic?: boolean;
  title?: string;
  toolMetadata?: RecordValue;
};

type StreamMetadata = {
  status: "success" | "failed";
  error?: string;
};

class OrphanToolInvocationError extends Error {}

/**
 * Recover stored messages from versioned AI SDK UIMessageChunk rows without
 * loading the AI SDK in the Convex component.
 *
 * UIMessageChunk is a persisted wire format here, not a core Agent type.
 * The legacy marker stays pinned to AI SDK 6 behavior. New v7-only chunks are
 * accepted only under the explicit UIMessageChunkV7 marker.
 */
export function materializeUIMessageChunks(
  stream: StreamMessage,
  chunks: readonly unknown[],
  metadata: StreamMetadata,
): MessageWithMetadataInternal[] {
  if (
    stream.format !== "UIMessageChunk" &&
    stream.format !== "UIMessageChunkV7"
  ) {
    throw new Error(
      `materializeUIMessageChunks: unsupported stream format "${stream.format ?? "text"}" for stream ${stream.streamId}`,
    );
  }

  const parts: LegacyUIMessagePart[] = [];
  const activeText: Record<string, TextPart> = {};
  const activeReasoning: Record<string, ReasoningPart> = {};
  const partialToolCalls: Record<string, PartialToolCall> = {};
  let isOrphanTolerantPrefix = true;

  const staticToolPart = (toolCallId: string) =>
    parts.find(
      (part): part is ToolPart =>
        isStaticToolPart(part) && part.toolCallId === toolCallId,
    );
  const dynamicToolPart = (toolCallId: string) =>
    parts.find(
      (part): part is ToolPart =>
        part.type === "dynamic-tool" && part.toolCallId === toolCallId,
    );
  const toolPart = (toolCallId: string) =>
    parts.find(
      (part): part is ToolPart =>
        isToolPart(part) && part.toolCallId === toolCallId,
    );
  const toolPartByApprovalId = (approvalId: string) =>
    parts.find(
      (part): part is ToolPart =>
        isToolPart(part) && part.approval?.id === approvalId,
    );

  const updateToolPart = (
    dynamic: boolean,
    options: Omit<ToolPart, "type"> & {
      toolName: string;
      setCallProviderMetadataOnExisting?: boolean;
    },
  ) => {
    const existing = dynamic
      ? dynamicToolPart(options.toolCallId)
      : staticToolPart(options.toolCallId);
    if (existing) {
      existing.state = options.state;
      existing.input = options.input;
      existing.rawInput = options.rawInput;
      existing.output = options.output;
      existing.errorText = options.errorText;
      existing.preliminary = options.preliminary;
      existing.providerExecuted =
        options.providerExecuted ?? existing.providerExecuted;
      if (options.title !== undefined) existing.title = options.title;
      if (options.toolMetadata !== undefined) {
        existing.toolMetadata = options.toolMetadata;
      }
      if (
        options.setCallProviderMetadataOnExisting &&
        options.callProviderMetadata !== undefined
      ) {
        existing.callProviderMetadata = options.callProviderMetadata;
      }
      if (options.resultProviderMetadata !== undefined) {
        existing.resultProviderMetadata = options.resultProviderMetadata;
      }
      if (dynamic) existing.toolName = options.toolName;
      return existing;
    }

    const created: ToolPart = {
      type: dynamic ? "dynamic-tool" : `tool-${options.toolName}`,
      toolName: dynamic ? options.toolName : undefined,
      toolCallId: options.toolCallId,
      state: options.state,
      input: options.input,
      rawInput: options.rawInput,
      output: options.output,
      errorText: options.errorText,
      preliminary: options.preliminary,
      providerExecuted: options.providerExecuted,
      callProviderMetadata: options.callProviderMetadata,
      resultProviderMetadata: options.resultProviderMetadata,
      title: options.title,
      toolMetadata: options.toolMetadata,
    };
    parts.push(created);
    return created;
  };

  try {
    for (const value of chunks) {
      const chunk = chunkRecord(value);
      switch (chunk.type) {
        case "text-start": {
          isOrphanTolerantPrefix = false;
          const part: TextPart = {
            type: "text",
            text: "",
            providerMetadata: providerMetadata(chunk.providerMetadata),
          };
          activeText[stringField(chunk, "id")] = part;
          parts.push(part);
          break;
        }
        case "text-delta": {
          const id = stringField(chunk, "id");
          const part = activeText[id];
          if (!part) {
            throw new Error(
              `Received text-delta for missing text part with ID "${id}".`,
            );
          }
          part.text += stringField(chunk, "delta");
          part.providerMetadata =
            providerMetadata(chunk.providerMetadata) ?? part.providerMetadata;
          break;
        }
        case "text-end": {
          const id = stringField(chunk, "id");
          const part = activeText[id];
          if (!part) {
            throw new Error(
              `Received text-end for missing text part with ID "${id}".`,
            );
          }
          part.providerMetadata =
            providerMetadata(chunk.providerMetadata) ?? part.providerMetadata;
          delete activeText[id];
          break;
        }
        case "reasoning-start": {
          isOrphanTolerantPrefix = false;
          const part: ReasoningPart = {
            type: "reasoning",
            text: "",
            providerMetadata: providerMetadata(chunk.providerMetadata),
          };
          activeReasoning[stringField(chunk, "id")] = part;
          parts.push(part);
          break;
        }
        case "reasoning-delta": {
          const id = stringField(chunk, "id");
          const part = activeReasoning[id];
          if (!part) {
            throw new Error(
              `Received reasoning-delta for missing reasoning part with ID "${id}".`,
            );
          }
          part.text += stringField(chunk, "delta");
          part.providerMetadata =
            providerMetadata(chunk.providerMetadata) ?? part.providerMetadata;
          break;
        }
        case "reasoning-end": {
          const id = stringField(chunk, "id");
          const part = activeReasoning[id];
          if (!part) {
            throw new Error(
              `Received reasoning-end for missing reasoning part with ID "${id}".`,
            );
          }
          part.providerMetadata =
            providerMetadata(chunk.providerMetadata) ?? part.providerMetadata;
          delete activeReasoning[id];
          break;
        }
        case "file": {
          isOrphanTolerantPrefix = false;
          const metadata = v7OnlyProviderMetadata(
            stream,
            chunk.providerMetadata,
            "file.providerMetadata",
          );
          parts.push({
            type: "file",
            url: stringField(chunk, "url"),
            mediaType: stringField(chunk, "mediaType"),
            providerMetadata: metadata,
          });
          break;
        }
        case "reasoning-file":
          requireV7Stream(stream, chunk.type);
          isOrphanTolerantPrefix = false;
          parts.push({
            type: "reasoning-file",
            url: stringField(chunk, "url"),
            mediaType: stringField(chunk, "mediaType"),
            providerMetadata: providerMetadata(chunk.providerMetadata),
          });
          break;
        case "custom":
          requireV7Stream(stream, chunk.type);
          isOrphanTolerantPrefix = false;
          parts.push({
            type: "custom",
            kind: stringField(chunk, "kind"),
            providerMetadata: providerMetadata(chunk.providerMetadata),
          });
          break;
        case "source-url":
          isOrphanTolerantPrefix = false;
          parts.push({
            type: "source-url",
            sourceId: stringField(chunk, "sourceId"),
            url: stringField(chunk, "url"),
            title: optionalString(chunk.title),
            providerMetadata: providerMetadata(chunk.providerMetadata),
          });
          break;
        case "source-document":
          isOrphanTolerantPrefix = false;
          parts.push({
            type: "source-document",
            sourceId: stringField(chunk, "sourceId"),
            mediaType: stringField(chunk, "mediaType"),
            title: stringField(chunk, "title"),
            filename: optionalString(chunk.filename),
            providerMetadata: providerMetadata(chunk.providerMetadata),
          });
          break;
        case "tool-input-start": {
          isOrphanTolerantPrefix = false;
          const toolCallId = stringField(chunk, "toolCallId");
          const toolName = stringField(chunk, "toolName");
          const dynamic = chunk.dynamic === true;
          const callProviderMetadata = v7OnlyProviderMetadata(
            stream,
            chunk.providerMetadata,
            "tool-input-start.providerMetadata",
          );
          partialToolCalls[toolCallId] = {
            text: "",
            toolName,
            dynamic,
            title: optionalString(chunk.title),
            toolMetadata: toolMetadata(stream, chunk.toolMetadata),
          };
          updateToolPart(dynamic, {
            toolCallId,
            toolName,
            state: "input-streaming",
            input: undefined,
            providerExecuted: optionalBoolean(chunk.providerExecuted),
            callProviderMetadata,
            title: optionalString(chunk.title),
            toolMetadata: toolMetadata(stream, chunk.toolMetadata),
          });
          break;
        }
        case "tool-input-delta": {
          const toolCallId = stringField(chunk, "toolCallId");
          const partial = partialToolCalls[toolCallId];
          if (!partial) {
            throw new Error(
              `Received tool-input-delta for missing tool call with ID "${toolCallId}".`,
            );
          }
          partial.text += stringField(chunk, "inputTextDelta");
          updateToolPart(partial.dynamic === true, {
            toolCallId,
            toolName: partial.toolName,
            state: "input-streaming",
            input: parseCompleteJson(partial.text),
            title: partial.title,
            toolMetadata: partial.toolMetadata,
          });
          break;
        }
        case "tool-input-available": {
          isOrphanTolerantPrefix = false;
          const dynamic = chunk.dynamic === true;
          updateToolPart(dynamic, {
            toolCallId: stringField(chunk, "toolCallId"),
            toolName: stringField(chunk, "toolName"),
            state: "input-available",
            input: chunk.input,
            providerExecuted: optionalBoolean(chunk.providerExecuted),
            callProviderMetadata: providerMetadata(chunk.providerMetadata),
            setCallProviderMetadataOnExisting: true,
            title: optionalString(chunk.title),
            toolMetadata: toolMetadata(stream, chunk.toolMetadata),
          });
          break;
        }
        case "tool-input-error": {
          isOrphanTolerantPrefix = false;
          const dynamic = chunk.dynamic === true;
          const metadata = providerMetadata(chunk.providerMetadata);
          updateToolPart(dynamic, {
            toolCallId: stringField(chunk, "toolCallId"),
            toolName: stringField(chunk, "toolName"),
            state: "output-error",
            input: dynamic ? chunk.input : undefined,
            rawInput: dynamic ? undefined : chunk.input,
            errorText: stringField(chunk, "errorText"),
            providerExecuted: optionalBoolean(chunk.providerExecuted),
            ...(stream.format === "UIMessageChunkV7"
              ? { resultProviderMetadata: metadata }
              : { callProviderMetadata: metadata }),
            title: optionalString(chunk.title),
            toolMetadata: toolMetadata(stream, chunk.toolMetadata),
          });
          break;
        }
        case "tool-approval-request": {
          if (
            chunk.isAutomatic !== undefined ||
            chunk.signature !== undefined
          ) {
            requireV7Stream(
              stream,
              "tool-approval-request.isAutomatic/signature",
            );
          }
          const toolCallId = stringField(chunk, "toolCallId");
          const invocation = toolPart(toolCallId);
          if (
            !invocation &&
            stream.format === "UIMessageChunkV7" &&
            isOrphanTolerantPrefix
          ) {
            continue;
          }
          const requiredInvocation = requireToolPart(invocation, toolCallId);
          requiredInvocation.state = "approval-requested";
          requiredInvocation.approval = {
            id: stringField(chunk, "approvalId"),
            isAutomatic: optionalBoolean(chunk.isAutomatic),
            signature: optionalString(chunk.signature),
          };
          break;
        }
        case "tool-approval-response": {
          requireV7Stream(stream, chunk.type);
          const approvalId = stringField(chunk, "approvalId");
          const invocation = toolPartByApprovalId(approvalId);
          if (
            !invocation &&
            stream.format === "UIMessageChunkV7" &&
            isOrphanTolerantPrefix
          ) {
            continue;
          }
          const requiredInvocation = requireToolPart(invocation, approvalId);
          requiredInvocation.state = "approval-responded";
          requiredInvocation.approval = {
            id: approvalId,
            approved: booleanField(chunk, "approved"),
            reason: optionalString(chunk.reason),
            isAutomatic: requiredInvocation.approval?.isAutomatic,
            signature: requiredInvocation.approval?.signature,
          };
          requiredInvocation.providerExecuted =
            optionalBoolean(chunk.providerExecuted) ??
            requiredInvocation.providerExecuted;
          requiredInvocation.callProviderMetadata =
            providerMetadata(chunk.providerMetadata) ??
            requiredInvocation.callProviderMetadata;
          break;
        }
        case "tool-output-denied": {
          const toolCallId = stringField(chunk, "toolCallId");
          const invocation = toolPart(toolCallId);
          if (
            !invocation &&
            stream.format === "UIMessageChunkV7" &&
            isOrphanTolerantPrefix
          ) {
            continue;
          }
          requireToolPart(invocation, toolCallId).state = "output-denied";
          break;
        }
        case "tool-output-available": {
          const toolCallId = stringField(chunk, "toolCallId");
          const resultProviderMetadata = v7OnlyProviderMetadata(
            stream,
            chunk.providerMetadata,
            "tool-output-available.providerMetadata",
          );
          const invocation = toolPart(toolCallId);
          if (
            !invocation &&
            stream.format === "UIMessageChunkV7" &&
            isOrphanTolerantPrefix
          ) {
            continue;
          }
          const requiredInvocation = requireToolPart(invocation, toolCallId);
          updateToolPart(requiredInvocation.type === "dynamic-tool", {
            toolCallId,
            toolName: getToolName(requiredInvocation),
            state: "output-available",
            input: requiredInvocation.input,
            output: chunk.output,
            preliminary: optionalBoolean(chunk.preliminary),
            providerExecuted: optionalBoolean(chunk.providerExecuted),
            resultProviderMetadata,
            title: requiredInvocation.title,
            toolMetadata:
              toolMetadata(stream, chunk.toolMetadata) ??
              requiredInvocation.toolMetadata,
          });
          break;
        }
        case "tool-output-error": {
          const toolCallId = stringField(chunk, "toolCallId");
          const resultProviderMetadata = v7OnlyProviderMetadata(
            stream,
            chunk.providerMetadata,
            "tool-output-error.providerMetadata",
          );
          const invocation = toolPart(toolCallId);
          if (
            !invocation &&
            stream.format === "UIMessageChunkV7" &&
            isOrphanTolerantPrefix
          ) {
            continue;
          }
          const requiredInvocation = requireToolPart(invocation, toolCallId);
          updateToolPart(requiredInvocation.type === "dynamic-tool", {
            toolCallId,
            toolName: getToolName(requiredInvocation),
            state: "output-error",
            input: requiredInvocation.input,
            rawInput: requiredInvocation.rawInput,
            errorText: stringField(chunk, "errorText"),
            providerExecuted: optionalBoolean(chunk.providerExecuted),
            resultProviderMetadata,
            title: requiredInvocation.title,
            toolMetadata:
              toolMetadata(stream, chunk.toolMetadata) ??
              requiredInvocation.toolMetadata,
          });
          break;
        }
        case "start-step":
          parts.push({ type: "step-start" });
          break;
        case "finish-step":
          for (const id of Object.keys(activeText)) delete activeText[id];
          for (const id of Object.keys(activeReasoning)) {
            delete activeReasoning[id];
          }
          break;
        case "error":
          throw new Error(stringField(chunk, "errorText"));
        case "start": {
          const messageId = optionalString(chunk.messageId);
          if (
            messageId !== undefined &&
            messageId !== `stream:${stream.streamId}`
          ) {
            throw new Error("Expecting to only make one UIMessage in a stream");
          }
          break;
        }
        case "finish":
        case "abort":
        case "message-metadata":
          break;
        default:
          if (chunk.type.startsWith("data-") && chunk.transient !== true) {
            isOrphanTolerantPrefix = false;
            const id = optionalString(chunk.id);
            const existing = id
              ? parts.find(
                  (part): part is DataPart =>
                    part.type === chunk.type && "id" in part && part.id === id,
                )
              : undefined;
            if (existing) {
              existing.data = chunk.data;
            } else {
              parts.push({
                type: chunk.type as `data-${string}`,
                id,
                data: chunk.data,
              });
            }
          } else if (!chunk.type.startsWith("data-")) {
            // Transient data parts are persisted but deliberately not
            // materialised; only genuinely unknown types are an error.
            throw new Error(
              `materializeUIMessageChunks: unsupported durable chunk type "${chunk.type}"`,
            );
          }
          break;
      }
    }
  } catch (error) {
    // AI SDK 6's recovery path deliberately tolerates a continuation stream
    // whose tool invocation was persisted in an earlier stream. It returns the
    // materialized prefix and ignores the remaining chunks.
    if (!(error instanceof OrphanToolInvocationError)) throw error;
    if (stream.format === "UIMessageChunkV7" && !isOrphanTolerantPrefix) {
      throw error;
    }
  }

  return partsToMessages(parts, stream, metadata);
}

export function getPersistedStreamParts(
  deltas: readonly StreamDelta[],
  fromCursor = 0,
): { parts: unknown[]; cursor: number } {
  const parts: unknown[] = [];
  let cursor = fromCursor;
  for (const delta of [...deltas].sort((a, b) => a.start - b.start)) {
    if (delta.parts.length === 0) {
      console.debug(`Got delta with no parts: ${JSON.stringify(delta)}`);
      continue;
    }
    if (cursor !== delta.start) {
      if (cursor >= delta.end) continue;
      if (cursor < delta.start) {
        console.warn(
          `Got delta for stream ${delta.streamId} that has a gap ${cursor} -> ${delta.start}`,
        );
        break;
      }
      throw new Error(
        `Got unexpected delta for stream ${delta.streamId}: delta: ${delta.start} -> ${delta.end} existing cursor: ${cursor}`,
      );
    }
    parts.push(...delta.parts);
    cursor = delta.end;
  }
  return { parts, cursor };
}

function partsToMessages(
  parts: LegacyUIMessagePart[],
  stream: StreamMessage,
  metadata: StreamMetadata,
): MessageWithMetadataInternal[] {
  const sources = parts
    .filter((part): part is SourcePart =>
      ["source-url", "source-document"].includes(part.type),
    )
    .map((part) =>
      part.type === "source-url"
        ? {
            type: "source" as const,
            sourceType: "url" as const,
            url: part.url,
            id: part.sourceId,
            providerMetadata: part.providerMetadata,
            title: part.title,
          }
        : {
            type: "source" as const,
            sourceType: "document" as const,
            mediaType: part.mediaType,
            id: part.sourceId,
            providerMetadata: part.providerMetadata,
            title: part.title,
            filename: part.filename,
          },
    );

  const blocks: LegacyUIMessagePart[][] = [];
  let block: LegacyUIMessagePart[] = [];
  const flush = () => {
    if (block.length > 0) blocks.push(block);
    block = [];
  };
  for (const part of parts) {
    if (part.type === "step-start") {
      flush();
    } else if (
      part.type === "text" ||
      part.type === "reasoning" ||
      part.type === "file" ||
      part.type === "reasoning-file" ||
      part.type === "custom" ||
      isToolPart(part) ||
      part.type.startsWith("data-")
    ) {
      block.push(part);
    }
  }
  flush();

  const messages: Message[] = [];
  for (const current of blocks) {
    const assistantContent: AssistantContentPart[] = [];
    const tools = current.filter(isToolPart);

    for (const part of current) {
      if (part.type === "text") {
        assistantContent.push({
          type: "text",
          text: part.text,
          ...(part.providerMetadata
            ? { providerOptions: part.providerMetadata }
            : {}),
        });
      } else if (part.type === "reasoning") {
        assistantContent.push({
          type: "reasoning",
          text: part.text,
          ...(part.providerMetadata
            ? { providerOptions: part.providerMetadata }
            : {}),
        });
      } else if (part.type === "file") {
        assistantContent.push({
          type: "file",
          data: part.url,
          filename: part.filename,
          mediaType: part.mediaType,
          ...(part.providerMetadata
            ? { providerOptions: part.providerMetadata }
            : {}),
        });
      } else if (part.type === "reasoning-file") {
        assistantContent.push({
          type: "reasoning-file",
          data: { type: "url", url: part.url },
          mediaType: part.mediaType,
          ...(part.providerMetadata
            ? { providerOptions: part.providerMetadata }
            : {}),
        });
      } else if (part.type === "custom") {
        assistantContent.push({
          type: "custom",
          kind: part.kind,
          ...(part.providerMetadata
            ? { providerOptions: part.providerMetadata }
            : {}),
        });
      } else if (isToolPart(part) && part.state !== "input-streaming") {
        const input =
          part.state === "output-error"
            ? (part.input ?? part.rawInput ?? {})
            : (part.input ?? {});
        assistantContent.push({
          type: "tool-call",
          toolCallId: part.toolCallId,
          toolName: getToolName(part),
          input,
          args: input,
          providerExecuted: part.providerExecuted,
          ...(stream.format === "UIMessageChunkV7" && part.title !== undefined
            ? { title: part.title }
            : {}),
          ...(stream.format === "UIMessageChunkV7" &&
          part.toolMetadata !== undefined
            ? { toolMetadata: part.toolMetadata }
            : {}),
          ...(part.callProviderMetadata
            ? { providerOptions: part.callProviderMetadata }
            : {}),
        });
        if (part.approval) {
          assistantContent.push({
            type: "tool-approval-request",
            approvalId: part.approval.id,
            toolCallId: part.toolCallId,
            isAutomatic: part.approval.isAutomatic,
            signature: part.approval.signature,
          });
        }
        if (
          part.providerExecuted === true &&
          part.state !== "approval-responded" &&
          (part.state === "output-available" || part.state === "output-error")
        ) {
          assistantContent.push(
            toolResult(
              part,
              part.state === "output-error" ? "error-json" : "normal",
              undefined,
              stream.format === "UIMessageChunk",
            ),
          );
        }
      }
    }
    if (assistantContent.length > 0) {
      messages.push({ role: "assistant", content: assistantContent });
    }

    const toolContent: ToolContentPart[] = [];
    for (const part of tools) {
      if (part.approval?.approved !== undefined) {
        toolContent.push({
          type: "tool-approval-response",
          approvalId: part.approval.id,
          approved: part.approval.approved,
          reason: part.approval.reason,
          providerExecuted: part.providerExecuted,
        });
      }
      if (part.providerExecuted === true) continue;
      if (part.state === "output-denied") {
        // Deliberately access approval like AI SDK 6: a denied chunk without a
        // preceding approval request is malformed rather than silently fixed.
        const reason = part.approval!.reason ?? "Tool execution denied.";
        toolContent.push(
          toolResult(
            part,
            "denied",
            reason,
            stream.format === "UIMessageChunk",
          ),
        );
      } else if (part.state === "output-error") {
        toolContent.push(
          toolResult(
            part,
            "error-text",
            undefined,
            stream.format === "UIMessageChunk",
          ),
        );
      } else if (part.state === "output-available") {
        toolContent.push(
          toolResult(
            part,
            "normal",
            undefined,
            stream.format === "UIMessageChunk",
          ),
        );
      }
    }
    if (toolContent.length > 0) {
      messages.push({ role: "tool", content: toolContent });
    }
  }

  return messages.map((message) => {
    const content = Array.isArray(message.content) ? message.content : [];
    const providerMetadataValue = content.find(
      (part) => part.providerOptions !== undefined,
    )?.providerOptions;
    const hasToolCall =
      message.role === "tool" ||
      content.some((part) => part.type === "tool-call");
    return {
      message,
      status: metadata.status,
      finishReason: hasToolCall ? "tool-calls" : "stop",
      model: stream.model,
      provider: stream.provider,
      ...(providerMetadataValue
        ? { providerMetadata: providerMetadataValue }
        : {}),
      sources,
      reasoning: content
        .filter(
          (part): part is Extract<MessageContentParts, { type: "reasoning" }> =>
            part.type === "reasoning",
        )
        .map((part) => part.text)
        .join(" "),
      ...(metadata.error !== undefined ? { error: metadata.error } : {}),
    } satisfies MessageWithMetadataInternal;
  });
}

function toolResult(
  part: ToolPart,
  mode: "normal" | "error-text" | "error-json" | "denied",
  deniedReason?: string,
  legacyCallMetadataFallback = false,
): Extract<MessageContentParts, { type: "tool-result" }> {
  const raw =
    mode === "denied"
      ? deniedReason
      : part.state === "output-error"
        ? part.errorText
        : part.output;
  const output =
    mode === "error-text" || mode === "denied"
      ? { type: "error-text" as const, value: String(raw) }
      : mode === "error-json"
        ? { type: "error-json" as const, value: raw ?? null }
        : typeof raw === "string"
          ? { type: "text" as const, value: raw }
          : { type: "json" as const, value: raw ?? null };
  const providerOptions =
    part.resultProviderMetadata ??
    (legacyCallMetadataFallback ? part.callProviderMetadata : undefined);
  return {
    type: "tool-result",
    toolCallId: part.toolCallId,
    toolName: getToolName(part),
    output,
    ...(!legacyCallMetadataFallback && part.providerExecuted !== undefined
      ? { providerExecuted: part.providerExecuted }
      : {}),
    ...(!legacyCallMetadataFallback && part.title !== undefined
      ? { title: part.title }
      : {}),
    ...(!legacyCallMetadataFallback && part.toolMetadata !== undefined
      ? { toolMetadata: part.toolMetadata }
      : {}),
    ...(providerOptions ? { providerOptions } : {}),
  };
}

function requireV7Stream(stream: StreamMessage, type: string): void {
  if (stream.format !== "UIMessageChunkV7") {
    throw new Error(
      `persisted chunk type "${type}" is not part of the pinned AI SDK 6.0.35 UIMessageChunk wire format`,
    );
  }
}

function chunkRecord(value: unknown): RecordValue & { type: string } {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid persisted UIMessageChunk");
  }
  const record = value as RecordValue;
  if (typeof record.type !== "string") {
    throw new Error("Persisted UIMessageChunk is missing a type");
  }
  return record as RecordValue & { type: string };
}

function stringField(record: RecordValue, field: string): string {
  const value = record[field];
  if (typeof value !== "string") {
    throw new Error(`Persisted UIMessageChunk field ${field} must be a string`);
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function optionalBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function booleanField(record: RecordValue, field: string): boolean {
  const value = record[field];
  if (typeof value !== "boolean") {
    throw new Error(
      `Persisted UIMessageChunk field ${field} must be a boolean`,
    );
  }
  return value;
}

function providerMetadata(value: unknown): ProviderMetadata | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as ProviderMetadata)
    : undefined;
}

function v7OnlyProviderMetadata(
  stream: StreamMessage,
  value: unknown,
  field: string,
): ProviderMetadata | undefined {
  if (value !== undefined) requireV7Stream(stream, field);
  return providerMetadata(value);
}

function optionalRecord(value: unknown): RecordValue | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as RecordValue)
    : undefined;
}

function toolMetadata(
  stream: StreamMessage,
  value: unknown,
): RecordValue | undefined {
  if (value === undefined) return undefined;
  requireV7Stream(stream, "toolMetadata");
  return optionalRecord(value);
}



function isToolPart(part: LegacyUIMessagePart): part is ToolPart {
  return part.type === "dynamic-tool" || part.type.startsWith("tool-");
}

function isStaticToolPart(part: LegacyUIMessagePart): part is ToolPart {
  return part.type.startsWith("tool-");
}

function getToolName(part: ToolPart): string {
  return part.type === "dynamic-tool"
    ? part.toolName!
    : part.type.split("-").slice(1).join("-");
}

function requireToolPart(
  part: ToolPart | undefined,
  toolCallId: string,
): ToolPart {
  if (!part) {
    throw new OrphanToolInvocationError(
      `No tool invocation found for tool call ID "${toolCallId}".`,
    );
  }
  return part;
}

function parseCompleteJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}
