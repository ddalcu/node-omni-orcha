#include "llm_binding.h"
#include "../common/async_helpers.h"
#include "../common/error_helpers.h"
#include "llama.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include <string>
#include <vector>
#include <cstring>
#include <functional>
#include <set>

// Inline helper — equivalent to common/common.h llama_batch_add
static void batch_add(
    struct llama_batch& batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool logits
) {
  batch.token   [batch.n_tokens] = id;
  batch.pos     [batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = (int32_t)seq_ids.size();
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
  }
  batch.logits  [batch.n_tokens] = logits ? 1 : 0;
  batch.n_tokens++;
}

// ─── Shared generation result ───

struct GenerationResult {
  std::string content;
  std::string reasoning_content;
  std::vector<common_chat_tool_call> tool_calls;
  int input_tokens = 0;
  int output_tokens = 0;
  std::string error;
};

// Token callback: called for each generated token piece. Return false to abort.
using token_callback_t = std::function<bool(const std::string& piece)>;

// ─── Core generation logic (shared by complete + stream) ───

static GenerationResult run_generation(
    llama_model* model,
    llama_context* ctx,
    const common_chat_templates* chat_templates,
    const std::vector<common_chat_msg>& messages,
    const std::vector<common_chat_tool>& tools,
    common_chat_tool_choice tool_choice,
    float temperature,
    int max_tokens,
    const std::vector<std::string>& stop_sequences,
    int thinking_budget,
    std::atomic<bool>& abort_flag,
    token_callback_t on_token)
{
  GenerationResult result;
  const llama_vocab* vocab = llama_model_get_vocab(model);

  // Build template inputs
  common_chat_templates_inputs inputs;
  inputs.messages = messages;
  inputs.add_generation_prompt = true;
  inputs.use_jinja = true;
  inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
  inputs.enable_thinking = true;

  bool has_tools = !tools.empty() && tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
  if (has_tools) {
    inputs.tools = tools;
    inputs.tool_choice = tool_choice;
  }

  // Apply chat template
  common_chat_params chat_params;
  try {
    chat_params = common_chat_templates_apply(chat_templates, inputs);
  } catch (const std::exception& e) {
    result.error = std::string("Failed to apply chat template: ") + e.what();
    return result;
  }

  const std::string& prompt = chat_params.prompt;

  // Merge stops
  auto all_stops = stop_sequences;
  for (const auto& s : chat_params.additional_stops) {
    all_stops.push_back(s);
  }

  // Tokenize
  int n_ctx = llama_n_ctx(ctx);
  std::vector<llama_token> tokens(n_ctx);
  int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                tokens.data(), tokens.size(), true, true);
  if (n_tokens < 0) {
    tokens.resize(-n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                              tokens.data(), tokens.size(), true, true);
  }
  if (n_tokens < 0) {
    result.error = "Tokenization failed";
    return result;
  }
  tokens.resize(n_tokens);
  result.input_tokens = n_tokens;

  // Clear KV cache
  llama_memory_t mem = llama_get_memory(ctx);
  if (mem) {
    llama_memory_clear(mem, true);
  }

  // Process prompt
  llama_batch batch = llama_batch_init(n_tokens, 0, 1);
  for (int i = 0; i < n_tokens; i++) {
    batch_add(batch, tokens[i], i, {0}, false);
  }
  batch.logits[batch.n_tokens - 1] = 1;

  if (llama_decode(ctx, batch) != 0) {
    llama_batch_free(batch);
    result.error = "Failed to process prompt";
    return result;
  }
  llama_batch_free(batch);

  // Build common_params_sampling from chat_params — same approach as llama-server.
  // This gives us all server features: grammar, reasoning budget, penalties, etc.
  common_params_sampling sparams;
  sparams.temp = temperature;
  sparams.generation_prompt = chat_params.generation_prompt;

  // Grammar from chat template (for tool calling)
  if (has_tools && !chat_params.grammar.empty()) {
    sparams.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat_params.grammar);
    sparams.grammar_lazy = chat_params.grammar_lazy;
    sparams.grammar_triggers = chat_params.grammar_triggers;
  }

  // Preserved tokens (special tokens to render as text, e.g. <think>, </think>)
  for (const auto& tok_str : chat_params.preserved_tokens) {
    auto ids = common_tokenize(vocab, tok_str, false, true);
    for (auto id : ids) {
      sparams.preserved_tokens.insert(id);
    }
  }

  // Reasoning budget
  if (thinking_budget >= 0 && !chat_params.thinking_end_tag.empty()) {
    sparams.reasoning_budget_tokens = thinking_budget;
    sparams.reasoning_budget_start = common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
    sparams.reasoning_budget_end   = common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
    sparams.reasoning_budget_forced = sparams.reasoning_budget_end;
  }

  // Use greedy sampling for temperature <= 0
  if (temperature <= 0.0f) {
    sparams.temp = 0.0f;
  }

  // Initialize sampler via common_sampler (same high-level API as llama-server)
  common_sampler* smpl = common_sampler_init(model, sparams);
  if (!smpl) {
    result.error = "Failed to initialize sampler";
    return result;
  }

  // Generate tokens
  std::string output;
  int cur_pos = n_tokens;

  for (int i = 0; i < max_tokens; i++) {
    if (abort_flag.load()) break;

    llama_token new_token = common_sampler_sample(smpl, ctx, -1);

    if (llama_vocab_is_eog(vocab, new_token)) break;

    common_sampler_accept(smpl, new_token, true);

    // Render token — preserved (special) tokens render as text, others filtered
    bool special = sparams.preserved_tokens.count(new_token) > 0;
    std::string piece = common_token_to_piece(vocab, new_token, special);
    if (!piece.empty()) {
      output.append(piece);
    }

    // Check stop sequences
    bool should_stop = false;
    for (const auto& stop : all_stops) {
      if (output.length() >= stop.length() &&
          output.compare(output.length() - stop.length(), stop.length(), stop) == 0) {
        output.resize(output.length() - stop.length());
        should_stop = true;
        break;
      }
    }

    // Emit token to callback (streaming)
    if (on_token && !piece.empty()) {
      if (!on_token(piece)) break;
    }

    if (should_stop) break;

    // Decode next
    llama_batch next_batch = llama_batch_init(1, 0, 1);
    batch_add(next_batch, new_token, cur_pos, {0}, true);
    cur_pos++;

    if (llama_decode(ctx, next_batch) != 0) {
      llama_batch_free(next_batch);
      common_sampler_free(smpl);
      result.error = "Decode failed during generation";
      return result;
    }
    llama_batch_free(next_batch);

    result.output_tokens++;
  }

  common_sampler_free(smpl);

  // Strip generation prompt from output if the model re-emitted it
  if (!output.empty() && !chat_params.generation_prompt.empty()) {
    const auto& gp = chat_params.generation_prompt;
    if (output.size() >= gp.size() && output.compare(0, gp.size(), gp) == 0) {
      output = output.substr(gp.size());
    }
  }

  // Parse output — extract content, reasoning, and tool calls via PEG parser
  if (!output.empty()) {
    try {
      common_chat_parser_params parser_params(chat_params);
      parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
      if (has_tools) {
        parser_params.parse_tool_calls = true;
      }
      if (!chat_params.parser.empty()) {
        parser_params.parser.load(chat_params.parser);
      }
      common_chat_msg parsed = common_chat_parse(output, false, parser_params);
      result.content = parsed.content;
      result.reasoning_content = parsed.reasoning_content;
      result.tool_calls = std::move(parsed.tool_calls);
    } catch (...) {
      result.content = output;
    }
  }

  return result;
}

// ─── Constructor ───

LlmContext::LlmContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<LlmContext>(info) {
  auto env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Model path (string) required");
  }
  if (info.Length() < 2 || !info[1].IsObject()) {
    throw Napi::TypeError::New(env, "Options object required");
  }

  std::string modelPath = info[0].As<Napi::String>().Utf8Value();
  Napi::Object opts = info[1].As<Napi::Object>();

  int contextSize = getInt32Option(opts, "contextSize", 4096);
  int gpuLayers = getInt32Option(opts, "gpuLayers", -1);
  bool flashAttn = getBoolOption(opts, "flashAttn", false);
  int batchSize = getInt32Option(opts, "batchSize", 512);
  std::string cacheTypeK = getStringOption(opts, "cacheTypeK", "f16");
  std::string cacheTypeV = getStringOption(opts, "cacheTypeV", "f16");

  llama_backend_init();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = gpuLayers;

  model_ = llama_model_load_from_file(modelPath.c_str(), model_params);
  if (!model_) {
    throw Napi::Error::New(env, "Failed to load model: " + modelPath);
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = contextSize;
  ctx_params.n_batch = batchSize;
  ctx_params.flash_attn_type = flashAttn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;

  if (cacheTypeK == "q8_0") ctx_params.type_k = GGML_TYPE_Q8_0;
  else if (cacheTypeK == "q4_0") ctx_params.type_k = GGML_TYPE_Q4_0;
  else ctx_params.type_k = GGML_TYPE_F16;

  if (cacheTypeV == "q8_0") ctx_params.type_v = GGML_TYPE_Q8_0;
  else if (cacheTypeV == "q4_0") ctx_params.type_v = GGML_TYPE_Q4_0;
  else ctx_params.type_v = GGML_TYPE_F16;

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    llama_model_free(model_);
    model_ = nullptr;
    throw Napi::Error::New(env, "Failed to create context");
  }

  n_ctx_ = contextSize;

  // Initialize Jinja chat templates from model metadata (or override)
  std::string chatTemplate = getStringOption(opts, "chatTemplate", "");
  chat_templates_ = common_chat_templates_init(model_, chatTemplate);
}

LlmContext::~LlmContext() {
  Cleanup();
}

void LlmContext::Cleanup() {
  chat_templates_.reset();
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

// ─── Message Parsing ───

std::vector<common_chat_msg> LlmContext::ParseMessages(const Napi::Array& arr) {
  std::vector<common_chat_msg> messages;
  messages.reserve(arr.Length());

  for (uint32_t i = 0; i < arr.Length(); i++) {
    Napi::Object msg = arr.Get(i).As<Napi::Object>();
    common_chat_msg m;
    m.role = msg.Get("role").As<Napi::String>().Utf8Value();
    m.content = msg.Get("content").As<Napi::String>().Utf8Value();

    if (msg.Has("tool_call_id") && msg.Get("tool_call_id").IsString()) {
      m.tool_call_id = msg.Get("tool_call_id").As<Napi::String>().Utf8Value();
    }
    if (msg.Has("name") && msg.Get("name").IsString()) {
      m.tool_name = msg.Get("name").As<Napi::String>().Utf8Value();
    }
    // Parse tool_calls array on assistant messages (for multi-turn)
    if (msg.Has("tool_calls") && msg.Get("tool_calls").IsArray()) {
      Napi::Array tcArr = msg.Get("tool_calls").As<Napi::Array>();
      for (uint32_t j = 0; j < tcArr.Length(); j++) {
        Napi::Object tc = tcArr.Get(j).As<Napi::Object>();
        common_chat_tool_call tool_call;
        tool_call.name = tc.Get("name").As<Napi::String>().Utf8Value();
        tool_call.arguments = tc.Get("args").As<Napi::String>().Utf8Value();
        if (tc.Has("id") && tc.Get("id").IsString()) {
          tool_call.id = tc.Get("id").As<Napi::String>().Utf8Value();
        }
        m.tool_calls.push_back(std::move(tool_call));
      }
    }

    messages.push_back(std::move(m));
  }

  return messages;
}

std::vector<common_chat_tool> LlmContext::ParseTools(const Napi::Array& arr) {
  std::vector<common_chat_tool> tools;
  tools.reserve(arr.Length());

  for (uint32_t i = 0; i < arr.Length(); i++) {
    Napi::Object t = arr.Get(i).As<Napi::Object>();
    common_chat_tool tool;
    tool.name = t.Get("name").As<Napi::String>().Utf8Value();
    tool.description = t.Get("description").As<Napi::String>().Utf8Value();
    // parameters is already JSON-serialized from TypeScript
    tool.parameters = t.Get("parameters").As<Napi::String>().Utf8Value();
    tools.push_back(std::move(tool));
  }

  return tools;
}

// ─── Helper: parse native options ───

struct NativeOpts {
  float temperature = 0.7f;
  int maxTokens = 2048;
  int thinkingBudget = -1;
  std::vector<std::string> stopSequences;
  std::vector<common_chat_tool> tools;
  common_chat_tool_choice toolChoice = COMMON_CHAT_TOOL_CHOICE_AUTO;
};

static NativeOpts parseNativeOpts(LlmContext* self, const Napi::Object& opts) {
  NativeOpts o;
  o.temperature = (float)getDoubleOption(opts, "temperature", 0.7);
  o.maxTokens = getInt32Option(opts, "maxTokens", 2048);
  o.thinkingBudget = getInt32Option(opts, "thinkingBudget", -1);

  if (opts.Has("stopSequences") && opts.Get("stopSequences").IsArray()) {
    Napi::Array stops = opts.Get("stopSequences").As<Napi::Array>();
    for (uint32_t i = 0; i < stops.Length(); i++) {
      o.stopSequences.push_back(stops.Get(i).As<Napi::String>().Utf8Value());
    }
  }

  if (opts.Has("tools") && opts.Get("tools").IsArray()) {
    o.tools = self->ParseTools(opts.Get("tools").As<Napi::Array>());
  }

  if (opts.Has("toolChoice") && opts.Get("toolChoice").IsString()) {
    std::string tc = opts.Get("toolChoice").As<Napi::String>().Utf8Value();
    if (tc == "required") o.toolChoice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    else if (tc == "none") o.toolChoice = COMMON_CHAT_TOOL_CHOICE_NONE;
    else o.toolChoice = COMMON_CHAT_TOOL_CHOICE_AUTO;
  }

  return o;
}

// ─── Helper: build Napi result object from GenerationResult ───

static Napi::Object resultToNapi(Napi::Env env, const GenerationResult& r) {
  Napi::Object result = Napi::Object::New(env);
  result.Set("content", Napi::String::New(env, r.content));

  if (!r.reasoning_content.empty()) {
    result.Set("reasoning", Napi::String::New(env, r.reasoning_content));
  }

  if (!r.tool_calls.empty()) {
    Napi::Array toolCalls = Napi::Array::New(env, r.tool_calls.size());
    for (size_t i = 0; i < r.tool_calls.size(); i++) {
      Napi::Object tc = Napi::Object::New(env);
      tc.Set("id", Napi::String::New(env, r.tool_calls[i].id));
      tc.Set("name", Napi::String::New(env, r.tool_calls[i].name));
      tc.Set("args", Napi::String::New(env, r.tool_calls[i].arguments));
      toolCalls.Set((uint32_t)i, tc);
    }
    result.Set("toolCalls", toolCalls);
  }

  Napi::Object usage = Napi::Object::New(env);
  usage.Set("inputTokens", Napi::Number::New(env, r.input_tokens));
  usage.Set("outputTokens", Napi::Number::New(env, r.output_tokens));
  usage.Set("totalTokens", Napi::Number::New(env, r.input_tokens + r.output_tokens));
  result.Set("usage", usage);

  return result;
}

// ─── Complete ───

class CompletionWorker : public Napi::AsyncWorker {
public:
  CompletionWorker(
    Napi::Env env,
    llama_model* model, llama_context* ctx,
    const common_chat_templates* chat_templates,
    std::vector<common_chat_msg> messages,
    NativeOpts opts,
    std::atomic<bool>& abort_flag
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      model_(model), ctx_(ctx),
      chat_templates_(chat_templates),
      messages_(std::move(messages)),
      opts_(std::move(opts)),
      abort_flag_(abort_flag) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    result_ = run_generation(
      model_, ctx_, chat_templates_,
      messages_, opts_.tools, opts_.toolChoice,
      opts_.temperature, opts_.maxTokens, opts_.stopSequences,
      opts_.thinkingBudget, abort_flag_, nullptr);
    if (!result_.error.empty()) {
      SetError(result_.error);
    }
  }

  void OnOK() override {
    deferred_.Resolve(resultToNapi(Env(), result_));
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  llama_model* model_;
  llama_context* ctx_;
  const common_chat_templates* chat_templates_;
  std::vector<common_chat_msg> messages_;
  NativeOpts opts_;
  std::atomic<bool>& abort_flag_;
  GenerationResult result_;
};

Napi::Value LlmContext::Complete(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (!ctx_ || !model_) throw Napi::Error::New(env, "Model not loaded");
  if (info.Length() < 1 || !info[0].IsArray()) throw Napi::TypeError::New(env, "Messages array required");

  auto messages = ParseMessages(info[0].As<Napi::Array>());
  NativeOpts opts;
  if (info.Length() >= 2 && info[1].IsObject()) {
    opts = parseNativeOpts(this, info[1].As<Napi::Object>());
  }

  abort_flag_.store(false);
  auto* worker = new CompletionWorker(env, model_, ctx_, chat_templates_.get(),
    std::move(messages), std::move(opts), abort_flag_);
  worker->Queue();
  return worker->Promise();
}

// ─── Stream ───

class StreamWorker : public Napi::AsyncWorker {
public:
  StreamWorker(
    Napi::Env env,
    llama_model* model, llama_context* ctx,
    const common_chat_templates* chat_templates,
    std::vector<common_chat_msg> messages,
    NativeOpts opts,
    std::atomic<bool>& abort_flag,
    Napi::Function callback
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      model_(model), ctx_(ctx),
      chat_templates_(chat_templates),
      messages_(std::move(messages)),
      opts_(std::move(opts)),
      abort_flag_(abort_flag)
  {
    tsfn_ = Napi::ThreadSafeFunction::New(env, callback, "llm-stream", 0, 1);
  }

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    // Token callback — sends each piece to JS via TSFN
    auto on_token = [this](const std::string& piece) -> bool {
      auto* data = new std::string(piece);
      auto status = tsfn_.BlockingCall(data, [](Napi::Env env, Napi::Function fn, std::string* text) {
        Napi::Object chunk = Napi::Object::New(env);
        chunk.Set("content", Napi::String::New(env, *text));
        chunk.Set("done", Napi::Boolean::New(env, false));
        fn.Call({chunk});
        delete text;
      });
      return status == napi_ok;
    };

    result_ = run_generation(
      model_, ctx_, chat_templates_,
      messages_, opts_.tools, opts_.toolChoice,
      opts_.temperature, opts_.maxTokens, opts_.stopSequences,
      opts_.thinkingBudget, abort_flag_, on_token);

    if (!result_.error.empty()) {
      SetError(result_.error);
    }
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);

    // Send final chunk with parsed reasoning, tool_calls, usage
    auto* data = new GenerationResult(result_);
    tsfn_.BlockingCall(data, [](Napi::Env env, Napi::Function fn, GenerationResult* r) {
      Napi::Object chunk = resultToNapi(env, *r);
      chunk.Set("done", Napi::Boolean::New(env, true));
      fn.Call({chunk});
      delete r;
    });

    tsfn_.Release();
    deferred_.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& e) override {
    tsfn_.Release();
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  Napi::ThreadSafeFunction tsfn_;
  llama_model* model_;
  llama_context* ctx_;
  const common_chat_templates* chat_templates_;
  std::vector<common_chat_msg> messages_;
  NativeOpts opts_;
  std::atomic<bool>& abort_flag_;
  GenerationResult result_;
};

Napi::Value LlmContext::Stream(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (!ctx_ || !model_) throw Napi::Error::New(env, "Model not loaded");
  if (info.Length() < 3 || !info[0].IsArray() || !info[2].IsFunction())
    throw Napi::TypeError::New(env, "stream(messages, opts, callback) required");

  auto messages = ParseMessages(info[0].As<Napi::Array>());
  Napi::Function callback = info[2].As<Napi::Function>();

  NativeOpts opts;
  if (info[1].IsObject()) {
    opts = parseNativeOpts(this, info[1].As<Napi::Object>());
  }

  abort_flag_.store(false);
  auto* worker = new StreamWorker(env, model_, ctx_, chat_templates_.get(),
    std::move(messages), std::move(opts), abort_flag_, callback);
  worker->Queue();
  return worker->Promise();
}

// ─── Embed ───

class EmbedWorker : public Napi::AsyncWorker {
public:
  EmbedWorker(
    Napi::Env env,
    llama_model* model,
    llama_context* ctx,
    std::string text
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      model_(model), ctx_(ctx),
      text_(std::move(text)) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    std::vector<llama_token> tokens(4096);
    int n_tokens = llama_tokenize(vocab, text_.c_str(), text_.length(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
      tokens.resize(-n_tokens);
      n_tokens = llama_tokenize(vocab, text_.c_str(), text_.length(),
                                tokens.data(), tokens.size(), true, true);
    }
    if (n_tokens < 0) {
      SetError("Tokenization failed for embedding");
      return;
    }
    tokens.resize(n_tokens);

    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem) {
      llama_memory_clear(mem, true);
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
      batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
    }

    if (llama_decode(ctx_, batch) != 0) {
      llama_batch_free(batch);
      SetError("Failed to compute embeddings");
      return;
    }

    int n_embd = llama_model_n_embd(model_);
    const float* embd = llama_get_embeddings_seq(ctx_, 0);
    if (!embd) {
      embd = llama_get_embeddings_ith(ctx_, -1);
    }

    if (embd) {
      embeddings_.resize(n_embd);
      for (int i = 0; i < n_embd; i++) {
        embeddings_[i] = (double)embd[i];
      }
    } else {
      SetError("Failed to get embeddings — model may not support embedding mode");
    }

    llama_batch_free(batch);
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);

    auto result = Napi::Float64Array::New(env, embeddings_.size());
    for (size_t i = 0; i < embeddings_.size(); i++) {
      result[i] = embeddings_[i];
    }

    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  llama_model* model_;
  llama_context* ctx_;
  std::string text_;
  std::vector<double> embeddings_;
};

Napi::Value LlmContext::Embed(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_ || !model_) {
    throw Napi::Error::New(env, "Model not loaded");
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Text string required");
  }

  std::string text = info[0].As<Napi::String>().Utf8Value();

  auto* worker = new EmbedWorker(env, model_, ctx_, std::move(text));
  worker->Queue();
  return worker->Promise();
}

Napi::Value LlmContext::EmbedBatch(const Napi::CallbackInfo& info) {
  throw Napi::Error::New(info.Env(), "Batch embedding not yet implemented");
}

// ─── Unload ───

Napi::Value LlmContext::Unload(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  Cleanup();
  auto deferred = Napi::Promise::Deferred::New(env);
  deferred.Resolve(env.Undefined());
  return deferred.Promise();
}

// ─── Abort ───

Napi::Value LlmContext::Abort(const Napi::CallbackInfo& info) {
  abort_flag_.store(true);
  return info.Env().Undefined();
}

// ─── Module ───

Napi::Value CreateContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  Napi::Function constructor = env.GetInstanceData<Napi::FunctionReference>()->Value();
  Napi::Object ctx = constructor.New({info[0], info[1]});
  auto deferred = Napi::Promise::Deferred::New(env);
  deferred.Resolve(ctx);
  return deferred.Promise();
}

Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
  Napi::Function func = LlmContext::DefineClass(env, "LlmContext", {
    LlmContext::InstanceMethod("complete", &LlmContext::Complete),
    LlmContext::InstanceMethod("stream", &LlmContext::Stream),
    LlmContext::InstanceMethod("embed", &LlmContext::Embed),
    LlmContext::InstanceMethod("embedBatch", &LlmContext::EmbedBatch),
    LlmContext::InstanceMethod("unload", &LlmContext::Unload),
    LlmContext::InstanceMethod("abort", &LlmContext::Abort),
  });

  auto* constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
  constructor->SuppressDestruct();
  env.SetInstanceData(constructor);

  exports.Set("createContext", Napi::Function::New(env, CreateContext));
  return exports;
}

NODE_API_MODULE(llm, InitModule)
