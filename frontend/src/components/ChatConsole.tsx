import { Fragment, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { Bot, ExternalLink, MessageSquare, Send, Settings2, Square, User } from "lucide-react";
import { CapabilityTransferRecord, InferenceTarget } from "../api/client";
import { InferenceDonePayload, streamInferenceRun } from "../api/sse";
import { ActiveTransferPill } from "./ActiveTransferPill";

type ChatRole = "user" | "assistant";
type ChatStatus = "done" | "streaming" | "error" | "stopped";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  status: ChatStatus;
  meta?: string;
};

type StreamStats = {
  startedAt: number;
  elapsedMs: number;
  tokenCount: number;
};

export function ChatConsole({
  activeInferenceTarget,
  capabilityTransfers,
  refresh,
  onToast
}: {
  activeInferenceTarget: InferenceTarget | null;
  capabilityTransfers: CapabilityTransferRecord[];
  refresh: () => void;
  onToast: (message: string, tone?: "info" | "success" | "error", title?: string) => void;
}) {
  const targetKey = useMemo(() => targetStorageKey(activeInferenceTarget), [activeInferenceTarget]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful model evaluation assistant. Answer clearly and stay faithful to the prompt.");
  const [temperature, setTemperature] = useState(0.2);
  const [topP, setTopP] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(128);
  const [stopText, setStopText] = useState("User:,System:");
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.08);
  const [noRepeatNgramSize, setNoRepeatNgramSize] = useState(3);
  const [doSample, setDoSample] = useState(true);
  const [asideOpen, setAsideOpen] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStats, setStreamStats] = useState<StreamStats | null>(null);
  const [error, setError] = useState("");
  const streamControllerRef = useRef<AbortController | null>(null);
  const activeAssistantIdRef = useRef<string | null>(null);
  const loadedHistoryKeyRef = useRef("");
  const skipNextHistorySaveRef = useRef(true);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    streamControllerRef.current?.abort();
    streamControllerRef.current = null;
    activeAssistantIdRef.current = null;
    setIsStreaming(false);
    setStreamStats(null);
    setError("");
    loadedHistoryKeyRef.current = targetKey;
    skipNextHistorySaveRef.current = true;
    setMessages(loadHistory(targetKey));
  }, [targetKey]);

  useEffect(() => {
    if (loadedHistoryKeyRef.current !== targetKey) {
      return;
    }
    if (skipNextHistorySaveRef.current) {
      skipNextHistorySaveRef.current = false;
      return;
    }
    saveHistory(targetKey, messages);
  }, [messages, targetKey]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    return () => {
      streamControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      return;
    }
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        stopStream();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isStreaming]);

  function submit(event: FormEvent) {
    event.preventDefault();
    const prompt = input.trim();
    if (!prompt || isStreaming || !activeInferenceTarget) {
      return;
    }

    const userMessage: ChatMessage = {
      id: makeId("user"),
      role: "user",
      content: prompt,
      createdAt: Date.now(),
      status: "done"
    };
    const assistantId = makeId("assistant");
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      status: "streaming"
    };
    const startedAt = performance.now();

    activeAssistantIdRef.current = assistantId;
    setInput("");
    setError("");
    setIsStreaming(true);
    setStreamStats({ startedAt, elapsedMs: 0, tokenCount: 0 });
    setMessages((current) => [...current, userMessage, assistantMessage].slice(-50));

    streamControllerRef.current = streamInferenceRun(
      {
        prompt,
        system: systemPrompt,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        stop: parseStopSequences(stopText),
        repetition_penalty: repetitionPenalty,
        no_repeat_ngram_size: noRepeatNgramSize,
        do_sample: doSample
      },
      {
        onToken: (token) => {
          updateAssistantMessage(assistantId, (message) => ({ ...message, content: message.content + token }));
          setStreamStats((current) => {
            const base = current || { startedAt, elapsedMs: 0, tokenCount: 0 };
            return {
              startedAt: base.startedAt,
              elapsedMs: performance.now() - base.startedAt,
              tokenCount: base.tokenCount + estimateTokenCount(token)
            };
          });
        },
        onDone: (payload) => finishAssistantMessage(assistantId, payload),
        onError: (message) => {
          setError(message);
          updateAssistantMessage(assistantId, (current) => ({
            ...current,
            status: "error",
            content: current.content || message,
            meta: "stream error"
          }));
        },
        onClose: () => {
          streamControllerRef.current = null;
          activeAssistantIdRef.current = null;
          setIsStreaming(false);
        }
      }
    );
  }

  function finishAssistantMessage(assistantId: string, payload: InferenceDonePayload) {
    const elapsedMs = typeof payload.elapsed_ms === "number" ? payload.elapsed_ms : streamStats?.elapsedMs;
    const tokens = typeof payload.tokens === "number" ? payload.tokens : streamStats?.tokenCount;
    updateAssistantMessage(assistantId, (message) => ({
      ...message,
      status: "done",
      meta: [tokens ? `${tokens} tokens` : "", elapsedMs ? `${Math.round(elapsedMs)} ms` : ""].filter(Boolean).join(" / ")
    }));
  }

  function stopStream() {
    streamControllerRef.current?.abort();
    const assistantId = activeAssistantIdRef.current;
    if (assistantId) {
      updateAssistantMessage(assistantId, (message) => ({
        ...message,
        status: message.status === "streaming" ? "stopped" : message.status,
        meta: message.status === "streaming" ? "stopped" : message.meta
      }));
    }
    setIsStreaming(false);
  }

  function updateAssistantMessage(messageId: string, updater: (message: ChatMessage) => ChatMessage) {
    setMessages((current) => current.map((message) => (message.id === messageId ? updater(message) : message)).slice(-50));
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  const elapsedSeconds = streamStats ? Math.max(streamStats.elapsedMs / 1000, 0.001) : 0;
  const tokensPerSecond = streamStats ? streamStats.tokenCount / elapsedSeconds : 0;
  const canSend = Boolean(input.trim() && activeInferenceTarget && !isStreaming);
  const baseCompletionTarget = isBaseCompletionTarget(activeInferenceTarget);

  return (
    <div className="thx thx-chat">
      <div className="thx-stage-h thx-chat-stage-h">
        <div>
          <div className="crumb">MORRIGAN · INFERENCE</div>
          <h2>Chat Console</h2>
          <p className="lede">Send prompts to the active inference target and inspect streamed output quality.</p>
        </div>
        <div className="thx-hud thx-chat-hud" aria-live="polite">
          <div><span className={isStreaming ? "thx-dot warn" : "thx-dot ok"} /> STATE <b>{isStreaming ? "STREAMING" : "READY"}</b></div>
          <div>LAT <b>{streamStats ? `${Math.round(streamStats.elapsedMs)} ms` : "--"}</b></div>
          <div>TPS <b>{streamStats ? tokensPerSecond.toFixed(1) : "--"}</b></div>
        </div>
      </div>

      <section className="thx-panel thx-panel--accent thx-chat-targetbar">
        <div className="thx-chat-target">
          <MessageSquare size={18} />
          <div>
            <span className={`thx-cap ${activeInferenceTarget ? "thx-cap--c" : "thx-cap--no"}`}>
              {activeInferenceTarget ? (activeInferenceTarget.target_type === "gguf_artifact" ? "GGUF" : "BASE") : "UNSET"}
            </span>
            <strong>{activeInferenceTarget?.display_name || "No active inference target"}</strong>
            <p>{targetDescription(activeInferenceTarget)}</p>
          </div>
        </div>
        <div className="thx-chat-target-actions">
          <ActiveTransferPill activeInferenceTarget={activeInferenceTarget} transfers={capabilityTransfers} refresh={refresh} onToast={onToast} />
          <button type="button" className="thx-btn" onClick={() => setAsideOpen((open) => !open)}>
            <Settings2 size={15} /> System
          </button>
          <a className="thx-btn" href="#/models">
            <ExternalLink size={15} /> Change in Models page
          </a>
        </div>
      </section>
      {baseCompletionTarget && (
        <div className="thx-chat-warning">
          <strong>BASE MODEL TARGET</strong>
          <span>Completion behavior can loop or continue the prompt. Prefer an Instruct target in Models for assistant-style chat.</span>
        </div>
      )}

      <div className={`thx-chat-layout ${asideOpen ? "" : "is-aside-collapsed"}`}>
        <section className="thx-panel thx-chat-panel">
          <div className="thx-panel-h">
            <div>
              <h3>Conversation</h3>
              <span className="thx-tag">[ HISTORY · {String(messages.length).padStart(2, "0")} ]</span>
            </div>
          </div>
          <div className="thx-chat-scroll" ref={scrollRef} aria-live="polite">
            {messages.length === 0 ? (
              <div className="thx-empty">NO MESSAGES FOR THIS TARGET</div>
            ) : (
              messages.map((message) => <ChatBubble key={message.id} message={message} />)
            )}
          </div>
          {error && <div className="thx-chat-error">{error}</div>}
          <form className="thx-chat-composer" onSubmit={submit}>
            <label className="thx-field thx-field--wide">
              <span className="thx-field-label">
                <span>Message</span>
                <span className="v">{input.length}</span>
              </span>
              <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={onComposerKeyDown} rows={4} />
            </label>
            <div className="thx-chat-controls">
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Temperature</span>
                  <span className="v">{temperature}</span>
                </span>
                <input type="number" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
              </label>
              <label className="thx-field thx-field--toggle">
                <span className="thx-field-label">
                  <span>Sampling</span>
                  <span className="v">{doSample ? "on" : "greedy"}</span>
                </span>
                <span className="thx-toggle">
                  <input type="checkbox" checked={doSample} onChange={(event) => setDoSample(event.target.checked)} />
                  <span className="thx-toggle-track" />
                </span>
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>top_p</span>
                  <span className="v">{topP}</span>
                </span>
                <input type="number" min="0.05" max="1" step="0.05" value={topP} onChange={(event) => setTopP(Number(event.target.value))} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Max tokens</span>
                  <span className="v">{maxTokens}</span>
                </span>
                <input type="number" min="1" max="4096" step="16" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Repeat penalty</span>
                  <span className="v">{repetitionPenalty}</span>
                </span>
                <input type="number" min="1" max="2" step="0.01" value={repetitionPenalty} onChange={(event) => setRepetitionPenalty(Number(event.target.value))} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>No repeat ngram</span>
                  <span className="v">{noRepeatNgramSize}</span>
                </span>
                <input type="number" min="0" max="12" step="1" value={noRepeatNgramSize} onChange={(event) => setNoRepeatNgramSize(Number(event.target.value))} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Stop</span>
                  <span className="v">{parseStopSequences(stopText).length}</span>
                </span>
                <input type="text" value={stopText} onChange={(event) => setStopText(event.target.value)} />
              </label>
            </div>
            <div className="thx-form-actions">
              {isStreaming ? (
                <button type="button" className="thx-btn thx-btn--danger" onClick={stopStream}>
                  <Square size={15} /> Stop
                </button>
              ) : (
                <button type="submit" className="thx-btn thx-btn--primary" disabled={!canSend}>
                  <Send size={15} /> Send
                </button>
              )}
            </div>
          </form>
        </section>

        {asideOpen && (
          <aside className="thx-panel thx-aside thx-chat-aside">
            <div className="thx-aside-h">
              <span>System Prompt</span>
              <span className="ping">editable</span>
            </div>
            <label className="thx-field thx-field--wide">
              <span className="thx-field-label">
                <span>System</span>
                <span className="v">{systemPrompt.length}</span>
              </span>
              <textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} rows={12} />
            </label>
            <div className="thx-aside-note">
              Active history is stored locally per inference target and capped at 50 messages.
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  return (
    <article className={`thx-chat-msg is-${message.role} is-${message.status}`}>
      <div className="thx-chat-msg-icon">{message.role === "user" ? <User size={15} /> : <Bot size={15} />}</div>
      <div className="thx-chat-msg-body">
        <div className="thx-chat-msg-meta">
          <span>{message.role}</span>
          <span>{new Date(message.createdAt).toLocaleTimeString()}</span>
          {message.meta && <span>{message.meta}</span>}
          {message.status === "streaming" && <span>streaming</span>}
          {message.status === "stopped" && <span>stopped</span>}
        </div>
        <div className="thx-chat-md">
          {message.content ? <Markdown content={message.content} /> : <span className="thx-chat-caret" />}
        </div>
      </div>
    </article>
  );
}

function Markdown({ content }: { content: string }) {
  return <>{renderMarkdownBlocks(content)}</>;
}

function renderMarkdownBlocks(content: string) {
  const blocks: JSX.Element[] = [];
  const codeFence = /```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = codeFence.exec(content)) !== null) {
    pushTextBlocks(blocks, content.slice(cursor, match.index), `t-${cursor}`);
    blocks.push(
      <pre className="thx-log thx-chat-code" key={`c-${match.index}`}>
        <code>{match[2]}</code>
      </pre>
    );
    cursor = match.index + match[0].length;
  }
  pushTextBlocks(blocks, content.slice(cursor), `t-${cursor}`);
  return blocks;
}

function pushTextBlocks(blocks: JSX.Element[], text: string, keyPrefix: string) {
  text.split(/\n{2,}/).forEach((paragraph, paragraphIndex) => {
    if (!paragraph.trim()) {
      return;
    }
    const lines = paragraph.split("\n");
    blocks.push(
      <p key={`${keyPrefix}-${paragraphIndex}`}>
        {lines.map((line, lineIndex) => (
          <Fragment key={`${keyPrefix}-${paragraphIndex}-${lineIndex}`}>
            {renderInlineMarkdown(line)}
            {lineIndex < lines.length - 1 && <br />}
          </Fragment>
        ))}
      </p>
    );
  });
}

function renderInlineMarkdown(text: string) {
  const nodes: Array<string | JSX.Element> = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*\n]+\*)/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index));
    }
    const value = match[0];
    if (value.startsWith("**")) {
      nodes.push(<strong key={`${match.index}-b`}>{value.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={`${match.index}-i`}>{value.slice(1, -1)}</em>);
    }
    cursor = match.index + value.length;
  }
  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }
  return nodes;
}

function targetStorageKey(target: InferenceTarget | null) {
  if (!target) {
    return "traininghub.chat.none";
  }
  const targetId = target.target_type === "gguf_artifact" ? target.artifact_id : target.model_slug;
  const transferId = target.capability_transfer_id || "no-transfer";
  return `traininghub.chat.${target.target_type}.${targetId || "active"}.${transferId}`;
}

function targetDescription(target: InferenceTarget | null) {
  if (!target) {
    return "Select a runtime before sending prompts.";
  }
  if (target.target_type === "gguf_artifact") {
    return target.path || target.artifact_id;
  }
  return target.provider_id || target.model_slug;
}

function isBaseCompletionTarget(target: InferenceTarget | null) {
  if (!target || target.target_type !== "base_model") {
    return false;
  }
  const text = `${target.model_slug} ${target.display_name} ${target.provider_id}`.toLowerCase();
  return /\bbase\b/.test(text) && !/\binstruct\b|\bit\b|chat/.test(text);
}

function loadHistory(storageKey: string): ChatMessage[] {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isChatMessage).slice(-50);
  } catch {
    return [];
  }
}

function saveHistory(storageKey: string, messages: ChatMessage[]) {
  window.localStorage.setItem(storageKey, JSON.stringify(messages.slice(-50)));
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Partial<ChatMessage>;
  return (
    typeof item.id === "string" &&
    (item.role === "user" || item.role === "assistant") &&
    typeof item.content === "string" &&
    typeof item.createdAt === "number" &&
    (item.status === "done" || item.status === "streaming" || item.status === "error" || item.status === "stopped")
  );
}

function parseStopSequences(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 8);
}

function estimateTokenCount(value: string) {
  const trimmed = value.trim();
  return trimmed ? Math.max(1, trimmed.split(/\s+/).length) : 0;
}

function makeId(prefix: string) {
  if ("randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
