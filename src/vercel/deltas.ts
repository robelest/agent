import {
  type DynamicToolUIPart,
  type ProviderMetadata,
  type ReasoningUIPart,
  type TextUIPart,
  type ToolUIPart,
  type UIMessageChunk,
} from "ai";
import { type UIMessage } from "./UIMessages.js";
import { joinText, sorted } from "../shared.js";
import {
  type MessageStatus,
  type StreamDelta,
  type StreamMessage,
} from "../validators.js";

export function blankUIMessage<METADATA = unknown>(
  streamMessage: StreamMessage & { metadata?: METADATA },
  threadId: string,
): UIMessage<METADATA> {
  return {
    id: `stream:${streamMessage.streamId}`,
    key: `${threadId}-${streamMessage.order}-${streamMessage.stepOrder}`,
    order: streamMessage.order,
    stepOrder: streamMessage.stepOrder,
    status: statusFromStreamStatus(streamMessage.status),
    agentName: streamMessage.agentName,
    text: "",
    _creationTime: Date.now(),
    role: "assistant",
    parts: [],
    ...(streamMessage.metadata ? { metadata: streamMessage.metadata } : {}),
  };
}

export function statusFromStreamStatus(
  status: StreamMessage["status"],
): MessageStatus | "streaming" {
  switch (status) {
    case "streaming":
      return "streaming";
    case "finished":
      return "success";
    case "aborted":
      return "failed";
    default:
      return "pending";
  }
}

export async function updateFromUIMessageChunks(
  uiMessage: UIMessage,
  parts: UIMessageChunk[],
  format: ReplayableUIMessageChunkFormat = "UIMessageChunk",
) {
  if (parts.length === 0) {
    return uiMessage;
  }
  return applyUIMessageChunksForFormat(
    uiMessage,
    parts,
    emptyIncrementalStreamState(),
    format,
  ).message;
}

type ToolPart = ToolUIPart | DynamicToolUIPart;

function transitionToolPart<S extends ToolPart["state"]>(
  part: ToolPart,
  updates: { state: S } & Partial<Extract<ToolPart, { state: S }>>,
): void {
  Object.assign(part, updates);
}

export type IncrementalStreamState = {
  // chunk id -> index of the streaming text part in message.parts
  activeText: Record<string, number>;
  // chunk id -> index of the streaming reasoning part in message.parts
  activeReasoning: Record<string, number>;
  // toolCallId -> raw accumulated input JSON text (kept separate from the
  // parsed `input` so partial JSON can be repair-parsed each batch)
  toolInputText: Record<string, string>;
};

export type ReplayableUIMessageChunkFormat =
  | "UIMessageChunk"
  | "UIMessageChunkV7";

export function emptyIncrementalStreamState(): IncrementalStreamState {
  return { activeText: {}, activeReasoning: {}, toolInputText: {} };
}

/** Apply AI SDK 7 chunks. New streams always use this path. */
export function applyUIMessageChunksIncremental(
  uiMessage: UIMessage,
  newParts: UIMessageChunk[],
  prev: IncrementalStreamState,
): { message: UIMessage; streamState: IncrementalStreamState } {
  return applyUIMessageChunksForFormat(
    uiMessage,
    newParts,
    prev,
    "UIMessageChunkV7",
  );
}

/**
 * Apply persisted AI SDK 6.0.35 chunks using the semantics of the version
 * that wrote them. This path must stay independent of the installed AI SDK so
 * upgrading the package cannot reinterpret existing streams.
 */
export function applyLegacyUIMessageChunksIncremental(
  uiMessage: UIMessage,
  newParts: UIMessageChunk[],
  prev: IncrementalStreamState,
): { message: UIMessage; streamState: IncrementalStreamState } {
  return applyUIMessageChunksForFormat(
    uiMessage,
    newParts,
    prev,
    "UIMessageChunk",
  );
}

export function applyPersistedUIMessageChunksIncremental(
  uiMessage: UIMessage,
  newParts: UIMessageChunk[],
  prev: IncrementalStreamState,
  format: StreamMessage["format"],
): { message: UIMessage; streamState: IncrementalStreamState } {
  switch (format) {
    case "UIMessageChunk":
      return applyLegacyUIMessageChunksIncremental(uiMessage, newParts, prev);
    case "UIMessageChunkV7":
      return applyUIMessageChunksIncremental(uiMessage, newParts, prev);
    default:
      throw new Error(
        `Unsupported UI message stream format "${format ?? "undefined"}"`,
      );
  }
}

/**
 * Apply a batch without replaying prior chunks. `prev` carries the ephemeral
 * state that the UIMessage itself cannot hold. The persisted format selects
 * the frozen wire semantics; callers must not infer it from chunk shape.
 */
function applyUIMessageChunksForFormat(
  uiMessage: UIMessage,
  newParts: UIMessageChunk[],
  prev: IncrementalStreamState,
  format: ReplayableUIMessageChunkFormat,
): { message: UIMessage; streamState: IncrementalStreamState } {
  const legacy = format === "UIMessageChunk";
  const message: UIMessage = structuredClone(uiMessage);
  const activeText: Record<string, number> = { ...prev.activeText };
  const activeReasoning: Record<string, number> = { ...prev.activeReasoning };
  const toolInputText: Record<string, string> = { ...prev.toolInputText };
  const touchedTools = new Set<string>();
  let hasLocallyAnchoredPart = false;

  const toolIndexById = new Map<string, number>();
  message.parts.forEach((p, i) => {
    if (
      "toolCallId" in p &&
      (p.type.startsWith("tool-") || p.type === "dynamic-tool")
    ) {
      toolIndexById.set((p as ToolPart).toolCallId, i);
    }
  });
  const toolPartAt = (toolCallId: string): ToolPart | undefined => {
    const idx = toolIndexById.get(toolCallId);
    return idx === undefined ? undefined : (message.parts[idx] as ToolPart);
  };
  const toolPartByApprovalId = (approvalId: string): ToolPart | undefined =>
    message.parts.find(
      (part): part is ToolPart =>
        (part.type.startsWith("tool-") || part.type === "dynamic-tool") &&
        "approval" in part &&
        part.approval?.id === approvalId,
    );
  const mergeMetadata = (metadata: unknown) => {
    if (metadata == null) {
      return;
    }
    message.metadata = {
      ...(message.metadata as Record<string, unknown> | undefined),
      ...(metadata as Record<string, unknown>),
    } as typeof message.metadata;
  };

  chunkLoop: for (const part of newParts) {
    if (legacy) assertLegacyUIMessageChunk(part);
    switch (part.type) {
      case "text-start": {
        hasLocallyAnchoredPart = true;
        const newPart: TextUIPart = {
          type: "text",
          text: "",
          state: "streaming",
          providerMetadata: part.providerMetadata,
        };
        message.parts.push(newPart);
        activeText[part.id] = message.parts.length - 1;
        break;
      }
      case "text-delta": {
        const idx = activeText[part.id];
        if (idx === undefined) {
          throw missingStreamPart("text-delta", part.id, "text");
        }
        const textPart = message.parts[idx] as TextUIPart;
        textPart.text += part.delta;
        textPart.providerMetadata =
          part.providerMetadata ?? textPart.providerMetadata;
        break;
      }
      case "text-end": {
        const idx = activeText[part.id];
        if (idx === undefined) {
          throw missingStreamPart("text-end", part.id, "text");
        }
        const textPart = message.parts[idx] as TextUIPart;
        textPart.state = "done";
        textPart.providerMetadata =
          part.providerMetadata ?? textPart.providerMetadata;
        delete activeText[part.id];
        break;
      }
      case "reasoning-start": {
        hasLocallyAnchoredPart = true;
        const newPart: ReasoningUIPart = {
          type: "reasoning",
          text: "",
          state: "streaming",
          providerMetadata: part.providerMetadata,
        };
        message.parts.push(newPart);
        activeReasoning[part.id] = message.parts.length - 1;
        break;
      }
      case "reasoning-delta": {
        const idx = activeReasoning[part.id];
        if (idx === undefined) {
          throw missingStreamPart("reasoning-delta", part.id, "reasoning");
        }
        const reasoningPart = message.parts[idx] as ReasoningUIPart;
        reasoningPart.text += part.delta;
        reasoningPart.providerMetadata =
          part.providerMetadata ?? reasoningPart.providerMetadata;
        break;
      }
      case "reasoning-end": {
        const idx = activeReasoning[part.id];
        if (idx === undefined) {
          throw missingStreamPart("reasoning-end", part.id, "reasoning");
        }
        const reasoningPart = message.parts[idx] as ReasoningUIPart;
        reasoningPart.state = "done";
        reasoningPart.providerMetadata =
          part.providerMetadata ?? reasoningPart.providerMetadata;
        delete activeReasoning[part.id];
        break;
      }
      case "tool-input-start": {
        hasLocallyAnchoredPart = true;
        const newToolPart: ToolUIPart | DynamicToolUIPart = part.dynamic
          ? ({
              type: "dynamic-tool",
              toolCallId: part.toolCallId,
              toolName: part.toolName,
              state: "input-streaming",
              input: undefined,
              providerExecuted: part.providerExecuted,
              callProviderMetadata: part.providerMetadata,
              title: part.title,
              toolMetadata: part.toolMetadata,
            } satisfies DynamicToolUIPart)
          : ({
              type: `tool-${part.toolName}`,
              toolCallId: part.toolCallId,
              state: "input-streaming",
              input: undefined,
              providerExecuted: part.providerExecuted,
              callProviderMetadata: part.providerMetadata,
              title: part.title,
              toolMetadata: part.toolMetadata,
            } satisfies ToolUIPart);
        message.parts.push(newToolPart);
        toolIndexById.set(part.toolCallId, message.parts.length - 1);
        toolInputText[part.toolCallId] = "";
        break;
      }
      case "tool-input-delta": {
        if (toolIndexById.has(part.toolCallId)) {
          toolInputText[part.toolCallId] =
            (toolInputText[part.toolCallId] ?? "") + part.inputTextDelta;
          touchedTools.add(part.toolCallId);
        } else {
          throw missingStreamPart(
            "tool-input-delta",
            part.toolCallId,
            "tool call",
          );
        }
        break;
      }
      case "tool-input-available": {
        hasLocallyAnchoredPart = true;
        let toolPart = toolPartAt(part.toolCallId);
        if (!toolPart) {
          toolPart = part.dynamic
            ? ({
                type: "dynamic-tool",
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                state: "input-available",
                input: part.input,
              } satisfies DynamicToolUIPart)
            : ({
                type: `tool-${part.toolName}`,
                toolCallId: part.toolCallId,
                state: "input-available",
                input: part.input,
              } satisfies ToolUIPart);
          message.parts.push(toolPart);
          toolIndexById.set(part.toolCallId, message.parts.length - 1);
        }
        if (toolPart) {
          transitionToolPart(toolPart, {
            state: "input-available",
            input: part.input,
            callProviderMetadata: legacy
              ? part.providerMetadata
              : (part.providerMetadata ??
                (toolPart as { callProviderMetadata?: ProviderMetadata })
                  .callProviderMetadata),
            title: part.title ?? toolPart.title,
            toolMetadata: part.toolMetadata ?? toolPart.toolMetadata,
            providerExecuted:
              part.providerExecuted ?? toolPart.providerExecuted,
          });
        }
        touchedTools.delete(part.toolCallId);
        // The raw JSON buffer is no longer needed; drop it so it doesn't get
        // carried through every later batch on the hot path.
        delete toolInputText[part.toolCallId];
        break;
      }
      case "tool-input-error": {
        hasLocallyAnchoredPart = true;
        let toolPart = toolPartAt(part.toolCallId);
        const created = toolPart === undefined;
        if (!toolPart) {
          toolPart = part.dynamic
            ? ({
                type: "dynamic-tool",
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                state: "output-error",
                input: part.input,
                errorText: part.errorText,
              } satisfies DynamicToolUIPart)
            : ({
                type: `tool-${part.toolName}`,
                toolCallId: part.toolCallId,
                state: "output-error",
                input: undefined,
                rawInput: part.input,
                errorText: part.errorText,
              } satisfies ToolUIPart);
          message.parts.push(toolPart);
          toolIndexById.set(part.toolCallId, message.parts.length - 1);
        }
        if (toolPart) {
          transitionToolPart(toolPart, {
            state: "output-error",
            errorText: part.errorText,
            ...(toolPart.type === "dynamic-tool"
              ? { input: part.input }
              : { input: undefined, rawInput: part.input }),
            ...(legacy
              ? {
                  callProviderMetadata: created
                    ? part.providerMetadata
                    : toolPart.callProviderMetadata,
                }
              : {
                  resultProviderMetadata:
                    part.providerMetadata ??
                    (
                      toolPart as {
                        resultProviderMetadata?: ProviderMetadata;
                      }
                    ).resultProviderMetadata,
                }),
            title: part.title ?? toolPart.title,
            toolMetadata: part.toolMetadata ?? toolPart.toolMetadata,
            providerExecuted:
              part.providerExecuted ?? toolPart.providerExecuted,
          });
        }
        touchedTools.delete(part.toolCallId);
        delete toolInputText[part.toolCallId];
        break;
      }
      case "tool-output-available": {
        const toolPart = toolPartAt(part.toolCallId);
        if (!toolPart && legacy) break chunkLoop;
        if (!toolPart && !hasLocallyAnchoredPart) continue chunkLoop;
        if (!toolPart) {
          throw missingStreamPart(
            "tool-output-available",
            part.toolCallId,
            "tool call",
          );
        }
        if (toolPart) {
          transitionToolPart(toolPart, {
            state: "output-available",
            output: part.output,
            preliminary: part.preliminary,
            providerExecuted:
              part.providerExecuted ?? toolPart.providerExecuted,
            resultProviderMetadata:
              part.providerMetadata ??
              (toolPart as { resultProviderMetadata?: ProviderMetadata })
                .resultProviderMetadata,
            toolMetadata: part.toolMetadata ?? toolPart.toolMetadata,
          });
        }
        break;
      }
      case "tool-output-error": {
        const toolPart = toolPartAt(part.toolCallId);
        if (!toolPart && legacy) break chunkLoop;
        if (!toolPart && !hasLocallyAnchoredPart) continue chunkLoop;
        if (!toolPart) {
          throw missingStreamPart(
            "tool-output-error",
            part.toolCallId,
            "tool call",
          );
        }
        if (toolPart) {
          transitionToolPart(toolPart, {
            state: "output-error",
            errorText: part.errorText,
            providerExecuted:
              part.providerExecuted ?? toolPart.providerExecuted,
            resultProviderMetadata:
              part.providerMetadata ??
              (toolPart as { resultProviderMetadata?: ProviderMetadata })
                .resultProviderMetadata,
            toolMetadata: part.toolMetadata ?? toolPart.toolMetadata,
          });
        }
        break;
      }
      case "tool-output-denied": {
        const toolPart = toolPartAt(part.toolCallId);
        if (!toolPart && legacy) break chunkLoop;
        if (!toolPart && !hasLocallyAnchoredPart) continue chunkLoop;
        if (!toolPart) {
          throw missingStreamPart(
            "tool-output-denied",
            part.toolCallId,
            "tool call",
          );
        }
        if (toolPart) {
          transitionToolPart(toolPart, { state: "output-denied" });
        }
        break;
      }
      case "tool-approval-request": {
        const toolPart = toolPartAt(part.toolCallId);
        if (!toolPart && legacy) break chunkLoop;
        if (!toolPart && !hasLocallyAnchoredPart) continue chunkLoop;
        if (!toolPart) {
          throw missingStreamPart(
            "tool-approval-request",
            part.toolCallId,
            "tool call",
          );
        }
        if (toolPart) {
          transitionToolPart(toolPart, {
            state: "approval-requested",
            approval: {
              id: part.approvalId,
              isAutomatic: legacy ? undefined : part.isAutomatic,
              signature: legacy ? undefined : part.signature,
            },
          });
        }
        break;
      }
      case "tool-approval-response": {
        const toolPart = toolPartByApprovalId(part.approvalId);
        if (!toolPart && !hasLocallyAnchoredPart) continue chunkLoop;
        if (!toolPart) {
          throw new Error(
            `No tool invocation found for approval ID "${part.approvalId}"`,
          );
        }
        const priorApproval = toolPart.approval;
        transitionToolPart(toolPart, {
          state: "approval-responded",
          input: toolPart.input,
          providerExecuted: part.providerExecuted ?? toolPart.providerExecuted,
          callProviderMetadata:
            part.providerMetadata ?? toolPart.callProviderMetadata,
          approval: {
            id: part.approvalId,
            approved: part.approved,
            reason: part.reason,
            isAutomatic: priorApproval?.isAutomatic,
            signature: priorApproval?.signature,
          },
        });
        break;
      }
      case "source-url":
        hasLocallyAnchoredPart = true;
        message.parts.push({
          type: "source-url",
          url: part.url,
          sourceId: part.sourceId,
          title: part.title,
          providerMetadata: part.providerMetadata,
        });
        break;
      case "source-document":
        hasLocallyAnchoredPart = true;
        message.parts.push({
          type: "source-document",
          mediaType: part.mediaType,
          sourceId: part.sourceId,
          title: part.title,
          filename: part.filename,
          providerMetadata: part.providerMetadata,
        });
        break;
      case "file":
        hasLocallyAnchoredPart = true;
        message.parts.push({
          type: "file",
          mediaType: part.mediaType,
          url: part.url,
          providerMetadata: legacy ? undefined : part.providerMetadata,
        });
        break;
      case "reasoning-file":
        hasLocallyAnchoredPart = true;
        message.parts.push({
          type: "reasoning-file",
          mediaType: part.mediaType,
          url: part.url,
          providerMetadata: part.providerMetadata,
        });
        break;
      case "custom":
        hasLocallyAnchoredPart = true;
        message.parts.push({
          type: "custom",
          kind: validateV7CustomKind(part.kind),
          providerMetadata: part.providerMetadata,
        });
        break;
      case "start-step":
        message.parts.push({ type: "step-start" });
        break;
      case "finish-step":
        // Match the SDK: a new step starts fresh streaming parts; the prior
        // parts keep their state rather than being forced to "done".
        for (const id of Object.keys(activeText)) delete activeText[id];
        for (const id of Object.keys(activeReasoning))
          delete activeReasoning[id];
        break;
      case "start":
        if (
          legacy &&
          part.messageId !== undefined &&
          part.messageId !== message.id
        ) {
          throw new Error("Expecting to only make one UIMessage in a stream");
        }
        mergeMetadata(part.messageMetadata);
        break;
      case "finish":
      case "message-metadata":
        mergeMetadata(part.messageMetadata);
        break;
      case "error":
        if (legacy) {
          message.status = "failed";
          break chunkLoop;
        }
        break;
      case "abort":
        // The stream-level status (statusFromStreamStatus) is authoritative and
        // is applied by the caller; nothing to mutate on the message here.
        break;
      default: {
        if (typeof part.type === "string" && part.type.startsWith("data-")) {
          hasLocallyAnchoredPart = true;
          const dataPart = part as Extract<
            UIMessageChunk,
            { type: `data-${string}` }
          >;
          if ("transient" in dataPart && dataPart.transient === true) break;
          const existingIdx =
            dataPart.id != null
              ? message.parts.findIndex(
                  (p) =>
                    p.type === dataPart.type &&
                    (p as { id?: string }).id === dataPart.id,
                )
              : -1;
          if (existingIdx >= 0) {
            (message.parts[existingIdx] as { data?: unknown }).data =
              dataPart.data;
          } else {
            message.parts.push(
              dataPart as unknown as UIMessage["parts"][number],
            );
          }
        } else {
          throw new Error(
            `applyUIMessageChunksIncremental: unsupported durable chunk type ${String(part.type)}`,
          );
        }
        break;
      }
    }
  }

  for (const toolCallId of touchedTools) {
    const toolPart = toolPartAt(toolCallId);
    if (toolPart && toolPart.state === "input-streaming") {
      try {
        toolPart.input = JSON.parse(toolInputText[toolCallId] ?? "");
      } catch {
        // partial JSON — leave input unset until complete
      }
    }
  }

  message.text = joinText(message.parts);
  return {
    message,
    streamState: { activeText, activeReasoning, toolInputText },
  };
}

function assertLegacyUIMessageChunk(part: UIMessageChunk): void {
  const record = part as unknown as Record<string, unknown>;
  const reject = () => {
    throw new Error(
      `persisted chunk type "${part.type}" is not part of the pinned AI SDK 6.0.35 UIMessageChunk wire format`,
    );
  };

  if (
    part.type === "custom" ||
    part.type === "reasoning-file" ||
    part.type === "tool-approval-response"
  ) {
    reject();
  }

  if ("toolMetadata" in record) reject();

  switch (part.type) {
    case "tool-input-start":
      if ("providerMetadata" in record) reject();
      break;
    case "file":
      if ("providerMetadata" in record) reject();
      break;
    case "tool-approval-request":
      if ("isAutomatic" in record || "signature" in record) reject();
      break;
    case "tool-output-available":
    case "tool-output-error":
      if ("providerMetadata" in record || "title" in record) reject();
      break;
  }
}

function validateV7CustomKind(kind: unknown): `${string}.${string}` {
  if (typeof kind !== "string") {
    throw new Error(
      `Invalid custom content kind; expected "{provider}.{provider-type}"`,
    );
  }
  const separator = kind.indexOf(".");
  if (separator <= 0 || separator === kind.length - 1) {
    throw new Error(
      `Invalid custom content kind "${kind}"; expected "{provider}.{provider-type}"`,
    );
  }
  return kind as `${string}.${string}`;
}

function missingStreamPart(
  chunkType: string,
  id: string,
  partType: string,
): Error {
  return new Error(
    `Received ${chunkType} for missing ${partType} with ID "${id}".`,
  );
}

export async function deriveUIMessagesFromDeltas(
  threadId: string,
  streamMessages: StreamMessage[],
  allDeltas: StreamDelta[],
): Promise<UIMessage[]> {
  const messages: UIMessage[] = [];
  for (const streamMessage of streamMessages) {
    if (
      streamMessage.format !== "UIMessageChunk" &&
      streamMessage.format !== "UIMessageChunkV7"
    ) {
      throw new Error(
        `deriveUIMessagesFromDeltas: unsupported stream format "${streamMessage.format ?? "text"}" for stream ${streamMessage.streamId}`,
      );
    }
    const { parts } = getParts<UIMessageChunk>(
      allDeltas.filter((d) => d.streamId === streamMessage.streamId),
      0,
    );
    const uiMessage = await updateFromUIMessageChunks(
      blankUIMessage(streamMessage, threadId),
      parts,
      streamMessage.format,
    );
    messages.push(uiMessage);
  }
  return sorted(messages);
}

export function getParts<T extends StreamDelta["parts"][number]>(
  deltas: StreamDelta[],
  fromCursor?: number,
): { parts: T[]; cursor: number } {
  const parts: T[] = [];
  let cursor = fromCursor ?? 0;
  for (const delta of deltas.sort((a, b) => a.start - b.start)) {
    if (delta.parts.length === 0) {
      console.debug(`Got delta with no parts: ${JSON.stringify(delta)}`);
      continue;
    }
    if (cursor !== delta.start) {
      if (cursor >= delta.end) {
        continue;
      } else if (cursor < delta.start) {
        console.warn(
          `Got delta for stream ${delta.streamId} that has a gap ${cursor} -> ${delta.start}`,
        );
        break;
      } else {
        throw new Error(
          `Got unexpected delta for stream ${delta.streamId}: delta: ${delta.start} -> ${delta.end} existing cursor: ${cursor}`,
        );
      }
    }
    parts.push(...delta.parts);
    cursor = delta.end;
  }
  return { parts, cursor };
}
