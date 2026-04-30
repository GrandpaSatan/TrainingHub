export type InferenceRunPayload = {
  prompt: string;
  system?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string[];
  repetition_penalty?: number;
  no_repeat_ngram_size?: number;
  do_sample?: boolean;
  dry_run?: boolean;
};

export type InferenceDonePayload = {
  elapsed_ms?: number;
  tokens?: number;
  target?: {
    target_type?: string;
    model_slug?: string;
    artifact_id?: string;
    display_name?: string;
  };
};

type SseHandlers = {
  onToken: (token: string) => void;
  onDone?: (payload: InferenceDonePayload) => void;
  onError?: (message: string) => void;
  onClose?: () => void;
};

type ParsedSseEvent = {
  event: string;
  data: Record<string, unknown>;
};

export function streamInferenceRun(payload: InferenceRunPayload, handlers: SseHandlers): AbortController {
  const controller = new AbortController();
  void readInferenceStream(payload, handlers, controller);
  return controller;
}

async function readInferenceStream(payload: InferenceRunPayload, handlers: SseHandlers, controller: AbortController) {
  try {
    const response = await fetch("/api/inference/run", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    if (!response.ok) {
      throw new Error(await responseText(response));
    }
    if (!response.body) {
      throw new Error("Inference stream did not return a readable response body.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let block = takeNextSseBlock(buffer);
      while (block) {
        buffer = block.rest;
        dispatchSseEvent(block.event, handlers);
        block = takeNextSseBlock(buffer);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      const event = parseSseBlock(buffer);
      if (event) {
        dispatchSseEvent(event, handlers);
      }
    }
  } catch (error) {
    if (!isAbortError(error)) {
      handlers.onError?.(error instanceof Error ? error.message : "Inference stream failed.");
    }
  } finally {
    handlers.onClose?.();
  }
}

function dispatchSseEvent(event: ParsedSseEvent, handlers: SseHandlers) {
  if (event.event === "token") {
    const token = event.data.token;
    if (typeof token === "string") {
      handlers.onToken(token);
    }
    return;
  }
  if (event.event === "done") {
    handlers.onDone?.(event.data as InferenceDonePayload);
    return;
  }
  if (event.event === "error") {
    const message = event.data.message;
    handlers.onError?.(typeof message === "string" ? message : "Inference failed.");
  }
}

function takeNextSseBlock(buffer: string): { event: ParsedSseEvent; rest: string } | null {
  const separator = /\r?\n\r?\n/.exec(buffer);
  if (!separator) {
    return null;
  }
  const rawBlock = buffer.slice(0, separator.index);
  const rest = buffer.slice(separator.index + separator[0].length);
  const event = parseSseBlock(rawBlock);
  return event ? { event, rest } : { event: { event: "message", data: {} }, rest };
}

function parseSseBlock(block: string): ParsedSseEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) as Record<string, unknown> };
  } catch {
    return { event, data: { value: dataLines.join("\n") } };
  }
}

async function responseText(response: Response) {
  try {
    const data = (await response.json()) as { detail?: string };
    return data.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
