// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }

      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

      export interface WireAnswer {
        text: string;
        toolCalls: Json[];
        stopReason: string;
      }

      export interface TokenUsage {
        input: number;
        output: number;
      }

      export interface ClientResponse {
        contentType: 'application/json' | 'text/event-stream; charset=utf-8';
        body: string;
      }

      function strings(value: unknown): string[] {
        if (typeof value === 'string') return [value];
        if (!Array.isArray(value)) return [];
        return value.filter((item): item is string => typeof item === 'string');
      }

      function imageDataUrl(block: Json): Json | null {
        if (block.type !== 'image' || !isRecord(block.source)) return null;
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
        }
        if (source.type === 'url' && typeof source.url === 'string') {
          return { type: 'image_url', image_url: { url: source.url } };
        }
        return null;
      }

      function openAIPartToGemini(part: unknown): Json | null {
        if (!isRecord(part)) return null;
        if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
        if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
        const data = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
        if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
        return { fileData: { mimeType: 'image/png', fileUri: part.image_url.url } };
      }

      function openAIContentParts(content: unknown): Json[] {
        if (typeof content === 'string') return [{ text: content }];
        if (!Array.isArray(content)) return [];
        return content.flatMap((part) => {
          const converted = openAIPartToGemini(part);
          return converted === null ? [] : [converted];
        });
      }

      function geminiFunctionChoice(choice: unknown): Json | null {
        if (choice === 'none') return { mode: 'NONE' };
        if (choice === 'required') return { mode: 'ANY' };
        if (choice === 'auto' || choice === undefined || choice === null) return { mode: 'AUTO' };
        if (!isRecord(choice)) return null;
        const fn = isRecord(choice.function) ? choice.function : {};
        if (typeof fn.name !== 'string') return null;
        return { mode: 'ANY', allowedFunctionNames: [fn.name] };
      }

      /** Translate an OpenAI chat request into Gemini generateContent JSON. */
      export function openAIToGemini(body: Json, model: string): Json {
        const contents: Json[] = [];
        const systemParts: Json[] = [];
        for (const raw of Array.isArray(body.messages) ? body.messages : []) {
          if (!isRecord(raw)) continue;
          if (raw.role === 'system' || raw.role === 'developer') {
            systemParts.push(...openAIContentParts(raw.content).filter((part) => typeof part.text === 'string'));
            continue;
          }
          if (raw.role === 'tool') {
            contents.push({
              role: 'function',
              parts: [{
                functionResponse: {
                  name: typeof raw.name === 'string' ? raw.name : '',
                  response: { content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') },
                },
              }],
            });
            continue;
          }
          const role = raw.role === 'assistant' ? 'model' : 'user';
          const parts = openAIContentParts(raw.content);
          for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            let args: unknown = {};
            try {
              args = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
            } catch {
              args = {};
            }
            parts.push({
              functionCall: {
                name: typeof call.function.name === 'string' ? call.function.name : '',
                args: isRecord(args) ? args : {},
              },
            });
          }
          if (parts.length > 0) contents.push({ role, parts });
        }

        const out: Json = { model, contents };
        if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
        const generationConfig: Json = {};
        if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
        if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
        if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
        if (Array.isArray(body.stop)) generationConfig.stopSequences = body.stop;
        if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
          generationConfig.responseMimeType = 'application/json';
        }
        if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
        if (Array.isArray(body.tools)) {
          const declarations = body.tools.flatMap((tool) => {
            if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') return [];
            return [{
              name: tool.function.name,
              description: typeof tool.function.description === 'string' ? tool.function.description : '',
              parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
            }];
          });
          if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
          const choice = geminiFunctionChoice(body.tool_choice);
          if (choice !== null) out.toolConfig = { functionCallingConfig: choice };
        }
        return out;
      }

      /** Translate an Anthropic Messages request into Gemini generateContent JSON. */
      export function anthropicToGemini(body: Json, model: string): Json {
        return openAIToGemini(anthropicToOpenAI(body, model), model);
      }

      /** Translate an OpenAI chat request into an Anthropic Messages API request. */
      export function openAIToAnthropicRequest(body: Json, model: string): Json {
        const messages: Json[] = [];
        const systemText: string[] = [];
        for (const raw of Array.isArray(body.messages) ? body.messages : []) {
          if (!isRecord(raw)) continue;
          if (raw.role === 'system' || raw.role === 'developer') {
            const text = textOf(raw.content);
            if (text !== '') systemText.push(text);
            continue;
          }
          if (raw.role === 'tool') {
            messages.push({
              role: 'user',
              content: [{
                type: 'tool_result',
                tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
                content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? ''),
              }],
            });
            continue;
          }
          const role = raw.role === 'assistant' ? 'assistant' : 'user';
          let content: Json[] = [];
          if (typeof raw.content === 'string') content.push({ type: 'text', text: raw.content });
          else if (Array.isArray(raw.content)) {
            content = raw.content.flatMap((part) => {
              if (!isRecord(part)) return [];
              if (part.type === 'text' && typeof part.text === 'string') return [{ type: 'text', text: part.text }];
              const image = imageDataUrl(part);
              if (image === null || !isRecord(image.image_url) || typeof image.image_url.url !== 'string') return [];
              const url = image.image_url.url;
              const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
              return data === null
                ? [{ type: 'image', source: { type: 'url', url } }]
                : [{ type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } }];
            });
          }
          for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            let input: unknown = {};
            try {
              input = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
            } catch {
              input = {};
            }
            content.push({
              type: 'tool_use',
              id: typeof call.id === 'string' ? call.id : '',
              name: typeof call.function.name === 'string' ? call.function.name : '',
              input: isRecord(input) ? input : {},
            });
          }
          messages.push({ role, content });
        }

        const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, messages };
        if (systemText.length > 0) out.system = systemText.join('\n');
        if (typeof body.temperature === 'number') out.temperature = body.temperature;
        if (typeof body.top_p === 'number') out.top_p = body.top_p;
        if (Array.isArray(body.stop)) out.stop_sequences = body.stop;
        if (Array.isArray(body.tools)) {
          out.tools = body.tools.flatMap((tool) => {
            if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') return [];
            return [{
              name: tool.function.name,
              description: typeof tool.function.description === 'string' ? tool.function.description : '',
              input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
            }];
          });
        }
        if (body.tool_choice === 'none') out.tool_choice = { type: 'none' };
        else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
        else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
          out.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
        } else if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
        if (body.stream === true) out.stream = true;
        return out;
      }

      /** Copy an Anthropic Messages request while selecting the routed model. */
      export function anthropicUpstreamRequest(body: Json, model: string): Json {
        return { ...body, model };
      }

      /** Ensure an OpenAI-wire upstream stream includes its final usage chunk. */
      export function withOpenAIStreamUsage(body: Json): Json {
        return {
          ...body,
          stream: true,
          stream_options: {
            ...(isRecord(body.stream_options) ? body.stream_options : {}),
            include_usage: true,
          },
        };
      }

      function providerAnswerFromGemini(value: Json): WireAnswer {
        const candidate = Array.isArray(value.candidates) && isRecord(value.candidates[0]) ? value.candidates[0] : {};
        const content = isRecord(candidate.content) ? candidate.content : {};
        const text: string[] = [];
        const toolCalls: Json[] = [];
        for (const part of Array.isArray(content.parts) ? content.parts : []) {
          if (!isRecord(part)) continue;
          if (typeof part.text === 'string') text.push(part.text);
          if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
            toolCalls.push({
              id: typeof part.functionCall.id === 'string' ? part.functionCall.id : `toolu_${toolCalls.length}`,
              type: 'function',
              function: {
                name: part.functionCall.name,
                arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
              },
            });
          }
        }
        return { text: text.join(''), toolCalls, stopReason: 'stop' };
      }

      function providerAnswerFromAnthropic(value: Json): WireAnswer {
        const text: string[] = [];
        const toolCalls: Json[] = [];
        for (const block of Array.isArray(value.content) ? value.content : []) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') text.push(block.text);
          if (block.type === 'tool_use') {
            toolCalls.push({
              id: typeof block.id === 'string' ? block.id : `toolu_${toolCalls.length}`,
              type: 'function',
              function: {
                name: typeof block.name === 'string' ? block.name : '',
                arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
              },
            });
          }
        }
        return {
          text: text.join(''),
          toolCalls,
          stopReason: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
        };
      }

      function framesOf(text: string): Array<{ event: string | null; data: string }> {
        const frames: Array<{ event: string | null; data: string }> = [];
        for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
          let event: string | null = null;
          const data: string[] = [];
          for (const line of frame.split('\n')) {
            if (line.startsWith('event:')) event = line.slice(6).trim();
            else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
          }
          if (data.length > 0) frames.push({ event, data: data.join('\n') });
        }
        return frames;
      }

      function providerAnswerFromGeminiStream(text: string): { answer: WireAnswer; usage: TokenUsage | null } {
        const answer: WireAnswer = { text: '', toolCalls: [], stopReason: 'stop' };
        let usage: TokenUsage | null = null;
        for (const frame of framesOf(text)) {
          let raw: unknown;
          try {
            raw = JSON.parse(frame.data);
          } catch {
            continue;
          }
          if (!isRecord(raw)) continue;
          const parsed = providerAnswerFromGemini(raw);
          answer.text += parsed.text;
          answer.toolCalls.push(...parsed.toolCalls);
          if (isRecord(raw.usageMetadata)) {
            const input = numberOrZero(raw.usageMetadata.promptTokenCount);
            const output = numberOrZero(raw.usageMetadata.candidatesTokenCount);
            if (input > 0 || output > 0) usage = { input, output };
          }
        }
        return { answer, usage };
      }

      function providerAnswerFromAnthropicStream(text: string): { answer: WireAnswer; usage: TokenUsage | null } {
        const answer: WireAnswer = { text: '', toolCalls: [], stopReason: 'end_turn' };
        const blocks = new Map<number, { type: string; id: string; name: string; json: string }>();
        let inputTokens: number | null = null;
        let outputTokens: number | null = null;
        for (const frame of framesOf(text)) {
          let raw: unknown;
          try {
            raw = JSON.parse(frame.data);
          } catch {
            continue;
          }
          if (!isRecord(raw)) continue;
          if (frame.event === 'message_start' && isRecord(raw.message) && isRecord(raw.message.usage)) {
            inputTokens = numberOrZero(raw.message.usage.input_tokens);
          } else if (frame.event === 'content_block_start') {
            const block = isRecord(raw.content_block) ? raw.content_block : {};
            const index = typeof raw.index === 'number' ? raw.index : blocks.size;
            blocks.set(index, {
              type: typeof block.type === 'string' ? block.type : '',
              id: typeof block.id === 'string' ? block.id : `toolu_${index}`,
              name: typeof block.name === 'string' ? block.name : '',
              json: JSON.stringify(isRecord(block.input) ? block.input : {}),
            });
          } else if (frame.event === 'content_block_delta' && isRecord(raw.delta)) {
            const index = typeof raw.index === 'number' ? raw.index : -1;
            const block = blocks.get(index);
            if (raw.delta.type === 'text_delta' && typeof raw.delta.text === 'string') answer.text += raw.delta.text;
            if (block !== undefined && raw.delta.type === 'input_json_delta' && typeof raw.delta.partial_json === 'string') {
              block.json = block.json === '{}' ? raw.delta.partial_json : block.json + raw.delta.partial_json;
            }
          } else if (frame.event === 'message_delta') {
            if (isRecord(raw.delta) && typeof raw.delta.stop_reason === 'string') answer.stopReason = raw.delta.stop_reason;
            if (isRecord(raw.usage)) outputTokens = numberOrZero(raw.usage.output_tokens);
          }
        }
        for (const block of blocks.values()) {
          if (block.type !== 'tool_use') continue;
          let args: unknown = {};
          try {
            args = JSON.parse(block.json);
          } catch {
            args = {};
          }
          answer.toolCalls.push({
            id: block.id,
            type: 'function',
            function: { name: block.name, arguments: JSON.stringify(isRecord(args) ? args : {}) },
          });
        }
        const usage = inputTokens === null && outputTokens === null
          ? null
          : { input: inputTokens ?? 0, output: outputTokens ?? 0 };
        return { answer, usage };
      }

      function requestTextAndTools(body: Json, clientWire: 'openai' | 'anthropic'): { system: string; user: string; tools: string[] } {
        if (clientWire === 'anthropic') {
          let user = '';
          for (const message of Array.isArray(body.messages) ? body.messages : []) {
            if (isRecord(message) && message.role === 'user') user = textOf(message.content);
          }
          const tools = Array.isArray(body.tools)
            ? body.tools.flatMap((tool) => (isRecord(tool) && typeof tool.name === 'string' ? [tool.name] : []))
            : [];
          return { system: textOf(body.system), user, tools };
        }
        let system = '';
        let user = '';
        for (const message of Array.isArray(body.messages) ? body.messages : []) {
          if (!isRecord(message)) continue;
          if (message.role === 'system') {
            const text = textOf(message.content);
            if (text !== '') system += `${system === '' ? '' : '\n'}${text}`;
          } else if (message.role === 'user') user = textOf(message.content);
        }
        const tools = Array.isArray(body.tools)
          ? body.tools.flatMap((tool) => (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string' ? [tool.function.name] : []))
          : [];
        return { system, user, tools };
      }

      /** Compute trial usage from request text and answer text when an upstream omits usage. */
      export function constructedUsage(body: Json, clientWire: 'openai' | 'anthropic', answerText: string): TokenUsage {
        const parts = requestTextAndTools(body, clientWire);
        return {
          input: 1 + Math.floor((parts.system.length + parts.user.length + parts.tools.join(',').length) / 4),
          output: 1 + Math.floor(answerText.length / 4),
        };
      }

      function geminiUsage(value: Json): TokenUsage | null {
        if (!isRecord(value.usageMetadata)) return null;
        const input = numberOrZero(value.usageMetadata.promptTokenCount);
        const output = numberOrZero(value.usageMetadata.candidatesTokenCount);
        return input > 0 || output > 0 ? { input, output } : null;
      }

      function anthropicUsage(value: Json): TokenUsage | null {
        if (!isRecord(value.usage)) return null;
        const input = numberOrZero(value.usage.input_tokens);
        const output = numberOrZero(value.usage.output_tokens);
        return input > 0 || output > 0 ? { input, output } : null;
      }

      function openAIStop(stopReason: string): string {
        if (stopReason === 'length' || stopReason === 'max_tokens') return 'length';
        if (stopReason === 'tool_calls' || stopReason === 'tool_use' || stopReason === 'function_call') return 'tool_calls';
        if (stopReason === 'content_filter' || stopReason === 'refusal') return 'content_filter';
        return 'stop';
      }

      function anthropicStop(stopReason: string): string {
        if (stopReason === 'length' || stopReason === 'max_tokens') return 'max_tokens';
        if (stopReason === 'tool_calls' || stopReason === 'tool_use' || stopReason === 'function_call') return 'tool_use';
        if (stopReason === 'content_filter' || stopReason === 'refusal') return 'refusal';
        return 'end_turn';
      }

      function clientOpenAI(answer: WireAnswer, usage: TokenUsage, model: string, stream: boolean): ClientResponse {
        const finishReason = openAIStop(answer.stopReason);
        const calls = answer.toolCalls.map((call, index) => ({
          id: typeof call.id === 'string' ? call.id : `toolu_${index}`,
          type: 'function',
          function: {
            name: isRecord(call.function) && typeof call.function.name === 'string' ? call.function.name : '',
            arguments: isRecord(call.function) && typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
          },
        }));
        const usageBody = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
        if (!stream) {
          return {
            contentType: 'application/json',
            body: JSON.stringify({
              id: 'chatcmpl-sim',
              object: 'chat.completion',
              created: FIXED_OPENAI_CREATED,
              model,
              choices: [{
                index: 0,
                message: { role: 'assistant', content: answer.text === '' ? null : answer.text, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
                finish_reason: finishReason,
              }],
              usage: usageBody,
            }),
          };
        }
        const frames: string[] = [];
        const chunk = (delta: Json, finish: string | null) => ({
          id: 'chatcmpl-sim',
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [{ index: 0, delta, finish_reason: finish }],
        });
        frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
        for (let at = 0; at < answer.text.length; at += 13) {
          frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + 13) }, null)));
        }
        for (const [index, call] of calls.entries()) {
          frames.push(sseEvent(null, chunk({
            tool_calls: [{ index, id: call.id, type: call.type, function: call.function }],
          }, null)));
        }
        frames.push(sseEvent(null, chunk({}, finishReason)));
        frames.push(sseEvent(null, {
          id: 'chatcmpl-sim',
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [],
          usage: usageBody,
        }));
        frames.push('data: [DONE]\n\n');
        return { contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
      }

      function clientAnthropic(answer: WireAnswer, usage: TokenUsage, model: string, stream: boolean): ClientResponse {
        const stopReason = anthropicStop(answer.stopReason);
        const content: Json[] = [];
        if (answer.text !== '') content.push({ type: 'text', text: answer.text });
        for (const call of answer.toolCalls) {
          const rawArguments = isRecord(call.function) && typeof call.function.arguments === 'string'
            ? call.function.arguments
            : '{}';
          let input: unknown = {};
          try {
            input = JSON.parse(rawArguments);
          } catch {
            input = {};
          }
          content.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : `toolu_${content.length}`,
            name: isRecord(call.function) && typeof call.function.name === 'string' ? call.function.name : '',
            input: isRecord(input) ? input : {},
          });
        }
        if (content.length === 0) content.push({ type: 'text', text: '' });
        if (!stream) {
          return {
            contentType: 'application/json',
            body: JSON.stringify({
              id: 'msg_sim',
              type: 'message',
              role: 'assistant',
              model,
              content,
              stop_reason: stopReason,
              stop_sequence: null,
              usage: { input_tokens: usage.input, output_tokens: usage.output },
            }),
          };
        }

        const frames = [sseEvent('message_start', {
          type: 'message_start',
          message: {
            id: 'msg_sim',
            type: 'message',
            role: 'assistant',
            model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: usage.input, output_tokens: 0 },
          },
        })];
        let index = 0;
        if (answer.text !== '') {
          frames.push(sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: { type: 'text', text: '' },
          }));
          for (let at = 0; at < answer.text.length; at += 13) {
            frames.push(sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index,
              delta: { type: 'text_delta', text: answer.text.slice(at, at + 13) },
            }));
          }
          frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
          index += 1;
        }
        for (const block of content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: block }));
          frames.push(sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) },
          }));
          frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
          index += 1;
        }
        if (index === 0) {
          frames.push(sseEvent('content_block_start', {
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'text', text: '' },
          }));
          frames.push(sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: '' },
          }));
          frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
        }
        frames.push(sseEvent('message_delta', {
          type: 'message_delta',
          delta: { stop_reason: stopReason, stop_sequence: null },
          usage: { output_tokens: usage.output },
        }));
        frames.push(sseEvent('message_stop', { type: 'message_stop' }));
        return { contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
      }

      /** Translate a Gemini or Anthropic upstream JSON response or complete SSE stream to either client wire. */
      export function upstreamToClient(
        upstreamWire: 'gemini' | 'anthropic',
        source: Json | string,
        clientWire: 'openai' | 'anthropic',
        request: Json,
        model: string,
        stream: boolean,
      ): ClientResponse {
        let answer: WireAnswer;
        let usage: TokenUsage | null;
        if (upstreamWire === 'gemini') {
          if (typeof source === 'string') {
            const parsed = providerAnswerFromGeminiStream(source);
            answer = parsed.answer;
            usage = parsed.usage;
          } else {
            answer = providerAnswerFromGemini(source);
            usage = geminiUsage(source);
          }
        } else if (typeof source === 'string') {
          const parsed = providerAnswerFromAnthropicStream(source);
          answer = parsed.answer;
          usage = parsed.usage;
        } else {
          answer = providerAnswerFromAnthropic(source);
          usage = anthropicUsage(source);
        }
        const computed = constructedUsage(request, clientWire, answer.text);
        const finalUsage = usage === null
          ? computed
          : { input: usage.input > 0 ? usage.input : computed.input, output: usage.output > 0 ? usage.output : computed.output };
        return clientWire === 'openai'
          ? clientOpenAI(answer, finalUsage, model, stream)
          : clientAnthropic(answer, finalUsage, model, stream);
      }
