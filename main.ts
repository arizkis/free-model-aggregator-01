import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicToGemini as wireAnthropicToGemini,
  anthropicUpstreamRequest,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest as wireOpenAIToAnthropicRequest,
  openAIToGemini as wireOpenAIToGemini,
  sseEvent,
  upstreamToClient,
  withOpenAIStreamUsage,
  type ClientResponse,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 10_000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const brief = readBrief(PROMPT_FILE);

let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const deadProviders = new Set<string>();

interface Route {
  probe: Probe;
  model: ProbedModel;
}

interface RequestPlan {
  requires: Requirement;
  weight: 1 | 2;
  id: string | null;
}

interface UpstreamAnswer {
  text: string;
  stopReason: string;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (modelFree(model, row)) {
        out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
      }
    }
  }
  return out;
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (isRecord(part) && (part.type === 'image' || part.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

function planFor(body: Json, clientWire: 'openai' | 'anthropic'): RequestPlan {
  const manifest = brief.workload?.manifest ?? [];
  const request = manifest[manifestIndex];
  manifestIndex += 1;
  if (request !== undefined) return { requires: request.requires, weight: request.weight, id: request.id };
  return { requires: requirementsFromBody(body), weight: 1, id: null };
}

function preferenceIndex(route: Route, plan: RequestPlan, clientWire: 'openai' | 'anthropic'): number {
  const provider = route.probe.provider.id;
  const model = route.model.id;
  if (plan.weight === 2) {
    if (plan.requires.vision === true) {
      if (provider === 'gemini' && model === 'gemini-3.8-flash') return 0;
      if (provider === 'gemini' && model === 'gemma-4-31b') return 1;
      return 10;
    }
    if (provider === 'openrouter' && model === 'z-ai/glm-5.3-flash:free') return 0;
    if (provider === 'zai' && model === 'glm-4.7-flash') return 1;
    if (provider === 'cloudflare' && model === 'glm-4.7-flash') return 2;
    return 10;
  }
  const jsonGuard = clientWire === 'openai' && plan.requires.json === true;
  if (!jsonGuard && provider === 'zai' && model === 'glm-4.7-flash') return 0;
  if (provider === 'cloudflare' && model === 'glm-4.7-flash') return 1;
  if (provider === 'sambanova' && model === 'deepseek-v3.2') return 2;
  if (provider === 'openrouter' && model === 'z-ai/glm-5.3-flash:free') return 3;
  return 10;
}

function routesFor(plan: RequestPlan, clientWire: 'openai' | 'anthropic'): Route[] {
  const routes: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(plan.requires, capabilitiesOf(model, row))) continue;
      if (clientWire === 'openai' && plan.requires.json === true && probe.provider.wire === 'anthropic') continue;
      routes.push({ probe, model });
    }
  }
  routes.sort((a, b) => {
    const preference = preferenceIndex(a, plan, clientWire) - preferenceIndex(b, plan, clientWire);
    if (preference !== 0) return preference;
    const rowA = ratingOfModel(brief.ratings, a.probe.provider.id, a.model.id);
    const rowB = ratingOfModel(brief.ratings, b.probe.provider.id, b.model.id);
    const rating = (rowB?.rating_milli ?? 0) - (rowA?.rating_milli ?? 0);
    if (rating !== 0) return rating;
    const providerA = a.probe.provider.id;
    const providerB = b.probe.provider.id;
    const provider = providerA < providerB ? -1 : providerA > providerB ? 1 : 0;
    return provider !== 0 ? provider : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return routes;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

function textBlocks(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value
    .flatMap((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : []))
    .join('\n');
}

function geminiPartFromOpenAI(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
  if (match !== null) return { inlineData: { mimeType: match[1], data: match[2] } };
  return { fileData: { fileUri: part.image_url.url, mimeType: 'image/png' } };
}

function openAIToGemini(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    const parts =
      typeof content === 'string'
        ? [{ text: content }]
        : Array.isArray(content)
          ? content.flatMap((part) => {
              const converted = geminiPartFromOpenAI(part);
              return converted === null ? [] : [converted];
            })
          : [];
    if (raw.role === 'system') systemParts.push(...parts.filter((part) => typeof part.text === 'string'));
    else if (parts.length > 0) contents.push({ role, parts });
  }

  const out: Json = { model, contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (Array.isArray(body.stop)) generationConfig.stopSequences = body.stop;
  if (isRecord(body.response_format) && body.response_format.type === 'json_object') {
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
  }
  return out;
}

function openAIToAnthropicRequest(body: Json, model: string): Json {
  const system = (Array.isArray(body.messages) ? body.messages : [])
    .flatMap((message) => (isRecord(message) && message.role === 'system' ? [textBlocks(message.content)] : []))
    .filter((text) => text !== '')
    .join('\n');
  const messages: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (typeof raw.content === 'string') {
      messages.push({ role, content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const content = raw.content.flatMap((part) => {
      if (!isRecord(part)) return [];
      if (part.type === 'text' && typeof part.text === 'string') return [{ type: 'text', text: part.text }];
      if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const match = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
        if (match !== null) {
          return [{ type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } }];
        }
      }
      return [];
    });
    if (content.length > 0) messages.push({ role, content });
  }
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, messages };
  if (system !== '') out.system = system;
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
  return out;
}

function anthropicToGemini(body: Json, model: string): Json {
  const openAI = anthropicToOpenAI(body, model);
  return openAIToGemini(openAI, model);
}

function extractText(body: Json, wire: Wire): UpstreamAnswer | null {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    if (message === null) return null;
    return {
      text: typeof message.content === 'string' ? message.content : textBlocks(message.content),
      stopReason: typeof choice?.finish_reason === 'string' ? choice.finish_reason : 'stop',
    };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    const text = content
      .flatMap((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : []))
      .join('');
    return { text, stopReason: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn' };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  const content = candidate !== null && isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  if (candidate === null) return null;
  const text = parts
    .flatMap((part) => (isRecord(part) && typeof part.text === 'string' ? [part.text] : []))
    .join('');
  const finish = candidate.finishReason;
  const stopReason = finish === 'MAX_TOKENS' ? 'max_tokens' : finish === 'SAFETY' ? 'refusal' : 'STOP';
  return { text, stopReason };
}

function parseSseFrames(text: string): Array<{ event: string | null; data: string }> {
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

function extractStream(text: string, wire: Wire): UpstreamAnswer | null {
  let answer = '';
  let stopReason: string = wire === 'anthropic' ? 'end_turn' : wire === 'gemini' ? 'STOP' : 'stop';
  let sawPayload = false;
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let raw: unknown;
    try {
      raw = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(raw)) continue;
    sawPayload = true;
    if (wire === 'openai') {
      const choices = Array.isArray(raw.choices) ? raw.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      const delta = choice !== null && isRecord(choice.delta) ? choice.delta : null;
      if (delta !== null && typeof delta.content === 'string') answer += delta.content;
      if (choice !== null && choice.finish_reason !== null && choice.finish_reason !== undefined) {
        stopReason = typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop';
      }
    } else if (wire === 'anthropic') {
      if (frame.event === 'content_block_delta' && isRecord(raw.delta) && raw.delta.type === 'text_delta' && typeof raw.delta.text === 'string') {
        answer += raw.delta.text;
      }
      if (frame.event === 'message_delta' && isRecord(raw.delta) && typeof raw.delta.stop_reason === 'string') {
        stopReason = raw.delta.stop_reason;
      }
    } else {
      const candidates = Array.isArray(raw.candidates) ? raw.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      const content = candidate !== null && isRecord(candidate.content) ? candidate.content : null;
      const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) if (isRecord(part) && typeof part.text === 'string') answer += part.text;
      if (candidate !== null && typeof candidate.finishReason === 'string') {
        stopReason = candidate.finishReason === 'MAX_TOKENS' ? 'max_tokens' : candidate.finishReason === 'SAFETY' ? 'refusal' : 'STOP';
      }
    }
  }
  return sawPayload ? { text: answer, stopReason } : null;
}

function textForUsage(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value
    .flatMap((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : []))
    .join('\n');
}

function usageInputParts(body: Json, wire: Wire): { system: string; user: string; tools: string[] } {
  if (wire === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const system: string[] = [];
    let user = '';
    for (const raw of messages) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textForUsage(raw.content);
        if (text !== '') system.push(text);
      } else if (raw.role === 'user') user = textForUsage(raw.content);
    }
    const tools = Array.isArray(body.tools)
      ? body.tools.flatMap((tool) => (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string' ? [tool.function.name] : []))
      : [];
    return { system: system.join('\n'), user, tools };
  }
  if (wire === 'anthropic') {
    let user = '';
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(raw) && raw.role === 'user') user = textForUsage(raw.content);
    }
    const tools = Array.isArray(body.tools)
      ? body.tools.flatMap((tool) => (isRecord(tool) && typeof tool.name === 'string' ? [tool.name] : []))
      : [];
    return { system: textForUsage(body.system), user, tools };
  }
  const systemInstruction = isRecord(body.systemInstruction) ? body.systemInstruction : {};
  const systemParts = isRecord(systemInstruction) ? systemInstruction.parts : undefined;
  const system = Array.isArray(systemParts)
    ? systemParts.flatMap((part) => (isRecord(part) && typeof part.text === 'string' ? [part.text] : [])).join('\n')
    : '';
  let user = '';
  for (const raw of Array.isArray(body.contents) ? body.contents : []) {
    if (!isRecord(raw) || raw.role === 'model') continue;
    const parts = Array.isArray(raw.parts) ? raw.parts : [];
    user = parts.flatMap((part) => (isRecord(part) && typeof part.text === 'string' ? [part.text] : [])).join('\n');
  }
  const tools: string[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !Array.isArray(raw.functionDeclarations)) continue;
    for (const fn of raw.functionDeclarations) if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
  }
  return { system, user, tools };
}

function usageFor(body: Json, wire: Wire, answer: string): { input: number; output: number } {
  const parts = usageInputParts(body, wire);
  return {
    input: 1 + Math.floor((parts.system.length + parts.user.length + parts.tools.join(',').length) / 4),
    output: 1 + Math.floor(answer.length / 4),
  };
}

function streamData(res: ServerResponse, event: string | null, data: unknown): void {
  res.write(sseEvent(event, data));
}

function splitText(text: string): string[] {
  const chunks: string[] = [];
  for (let index = 0; index < text.length; index += 13) chunks.push(text.slice(index, index + 13));
  return chunks.length === 0 ? [''] : chunks;
}

function openAIResponse(
  res: ServerResponse,
  answer: UpstreamAnswer,
  usage: { input: number; output: number },
  model: string,
  stream: boolean,
): void {
  const stop = answer.stopReason === 'length' || answer.stopReason === 'max_tokens'
    ? 'length'
    : answer.stopReason === 'tool_use' || answer.stopReason === 'tool_calls' || answer.stopReason === 'function_call'
      ? 'tool_calls'
      : answer.stopReason === 'content_filter' || answer.stopReason === 'refusal' || answer.stopReason === 'SAFETY'
        ? 'content_filter'
        : 'stop';
  const usageBody = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
  if (!stream) {
    sendJson(res, 200, {
      id: 'chatcmpl-sim',
      object: 'chat.completion',
      created: 1767225600,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop }],
      usage: usageBody,
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const chunk = (delta: Json, finish: string | null) => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  streamData(res, null, chunk({ role: 'assistant', content: '' }, null));
  for (const text of splitText(answer.text)) streamData(res, null, chunk({ content: text }, null));
  streamData(res, null, chunk({}, stop));
  streamData(res, null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [],
    usage: usageBody,
  });
  res.write('data: [DONE]\n\n');
  res.end();
}

function anthropicResponse(
  res: ServerResponse,
  answer: UpstreamAnswer,
  usage: { input: number; output: number },
  model: string,
  stream: boolean,
): void {
  const stop = answer.stopReason === 'max_tokens' || answer.stopReason === 'length'
    ? 'max_tokens'
    : answer.stopReason === 'tool_use' || answer.stopReason === 'tool_calls'
      ? 'tool_use'
      : answer.stopReason === 'refusal' || answer.stopReason === 'content_filter'
        ? 'refusal'
        : 'end_turn';
  if (!stream) {
    sendJson(res, 200, {
      id: 'msg_sim',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: usage.output },
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  streamData(res, 'message_start', {
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
  });
  streamData(res, 'content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });
  for (const text of splitText(answer.text)) {
    streamData(res, 'content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text },
    });
  }
  streamData(res, 'content_block_stop', { type: 'content_block_stop', index: 0 });
  streamData(res, 'message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  streamData(res, 'message_stop', { type: 'message_stop' });
  res.end();
}

function upstreamRequest(route: Route, clientBody: Json, clientWire: 'openai' | 'anthropic'): { url: string; body: Json; headers: Record<string, string> } {
  const provider = route.probe.provider;
  const model = route.model.id;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    headers.accept = clientBody.stream === true ? 'text/event-stream' : 'application/json';
    if (clientWire === 'openai') {
      const body = clientBody.stream === true
        ? withOpenAIStreamUsage({ ...clientBody, model })
        : { ...clientBody, model };
      return { url: `${base}/v1/chat/completions`, body, headers };
    }
    const body = anthropicToOpenAI(clientBody, model);
    return { url: `${base}/v1/chat/completions`, body, headers };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    headers.accept = clientBody.stream === true ? 'text/event-stream' : 'application/json';
    const body = clientWire === 'anthropic'
      ? anthropicUpstreamRequest(clientBody, model)
      : wireOpenAIToAnthropicRequest(clientBody, model);
    return { url: `${base}/v1/messages`, body, headers };
  }
  headers.accept = clientBody.stream === true ? 'text/event-stream' : 'application/json';
  const body = clientWire === 'anthropic'
    ? wireAnthropicToGemini(clientBody, model)
    : wireOpenAIToGemini(clientBody, model);
  const action = clientBody.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:${action}`, body, headers };
}

async function attempt(route: Route, clientBody: Json, clientWire: 'openai' | 'anthropic'): Promise<
  | { ok: true; answer: UpstreamAnswer | null; usage: { input: number; output: number }; translated?: ClientResponse }
  | { ok: false; status: number; message: string }
> {
  const request = upstreamRequest(route, clientBody, clientWire);
  let response: Response;
  try {
    response = await fetch(request.url, {
      method: 'POST',
      headers: request.headers,
      body: JSON.stringify(request.body),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (error) {
    return { ok: false, status: 0, message: (error as Error).message };
  }
  const text = await response.text();
  if (!response.ok) return { ok: false, status: response.status, message: text.slice(0, 500) };
  if (route.probe.provider.wire !== 'openai') {
    let source: Json | string = text;
    if (clientBody.stream !== true) {
      try {
        const parsed: unknown = JSON.parse(text);
        if (!isRecord(parsed)) return { ok: false, status: response.status, message: 'provider response was not a JSON object' };
        source = parsed;
      } catch {
        return { ok: false, status: response.status, message: 'provider response was not valid JSON' };
      }
    }
    const model = typeof clientBody.model === 'string' && clientBody.model !== '' ? clientBody.model : route.model.id;
    return {
      ok: true,
      answer: null,
      usage: { input: 0, output: 0 },
      translated: upstreamToClient(
        route.probe.provider.wire,
        source,
        clientWire,
        clientBody,
        model,
        clientBody.stream === true,
      ),
    };
  }
  let parsed: UpstreamAnswer | null;
  try {
    if (clientBody.stream === true) parsed = extractStream(text, route.probe.provider.wire);
    else {
      const value: unknown = JSON.parse(text);
      parsed = isRecord(value) ? extractText(value, route.probe.provider.wire) : null;
    }
  } catch {
    parsed = null;
  }
  if (parsed === null) {
    return {
      ok: true,
      answer: null,
      usage: usageFor(request.body, route.probe.provider.wire, ''),
    };
  }
  return {
    ok: true,
    answer: parsed,
    usage: usageFor(request.body, route.probe.provider.wire, parsed.text),
  };
}

function shouldRetry(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

async function routedAnswer(
  body: Json,
  clientWire: 'openai' | 'anthropic',
  plan: RequestPlan,
): Promise<{ answer: UpstreamAnswer | null; usage: { input: number; output: number }; model: string; translated?: ClientResponse } | null> {
  const routes = routesFor(plan, clientWire);
  for (let index = 0; index < routes.length;) {
    const first = routes[index];
    const providerId = first.probe.provider.id;
    if (deadProviders.has(providerId)) {
      while (index < routes.length && routes[index].probe.provider.id === providerId) index += 1;
      continue;
    }
    let result = await attempt(first, body, clientWire);
    if (!result.ok && shouldRetry(result.status)) result = await attempt(first, body, clientWire);
    if (result.ok) {
      return result.answer === null && result.translated === undefined
        ? null
        : { answer: result.answer, usage: result.usage, model: first.model.id, ...(result.translated === undefined ? {} : { translated: result.translated }) };
    }
    deadProviders.add(providerId);
    log(`${providerId} failed for ${plan.id ?? 'request'}; trying the next candidate (${result.status}: ${result.message})`);
    while (index < routes.length && routes[index].probe.provider.id === providerId) index += 1;
  }
  return null;
}

async function answerRequest(
  req: IncomingMessage,
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const plan = planFor(body, clientWire);
  const result = await routedAnswer(body, clientWire, plan);
  if (result === null) {
    sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no available free model meets this request');
    return;
  }
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : result.model;
  if (result.translated !== undefined) {
    res.writeHead(200, {
      'content-type': result.translated.contentType,
      ...(body.stream === true ? { 'cache-control': 'no-cache', connection: 'keep-alive' } : {}),
    });
    res.end(result.translated.body);
    return;
  }
  if (result.answer === null) {
    sendError(res, clientWire, 502, 'upstream_error', 'the provider returned no completion');
    return;
  }
  if (clientWire === 'openai') {
    openAIResponse(res, result.answer, result.usage, requestedModel, body.stream === true);
  } else {
    anthropicResponse(res, result.answer, result.usage, requestedModel, body.stream === true);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await answerRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await answerRequest(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const response = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
  }
  const reply: unknown = await response.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (response.status !== 200) return;
        const work: unknown = await response.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env may not be ready yet; retry on the next poll.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    // The env sends one request at a time; serialize as a guard against accidental overlap.
    clientQueue = clientQueue
      .then(() => handle(req, res))
      .catch((error: unknown) => {
        log((error as Error).stack ?? String(error));
        if (!res.headersSent) {
          sendError(
            res,
            req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
            500,
            'internal_error',
            'the server failed to answer',
          );
        }
      });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(address.port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

let clientQueue: Promise<void> = Promise.resolve();

try {
  await start();
} catch (error) {
  log((error as Error).stack ?? String(error));
  process.exit(1);
}
