# Tool Calling

vllm-mlx supports OpenAI-compatible tool calling (function calling) with automatic parsing for many popular model families.

## Quick Start

Enable tool calling by adding the `--enable-auto-tool-choice` flag when starting the server:

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Then use tools with the standard OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
```

## Streaming argument recovery

Chat Completions streams parser-produced tool argument fragments unchanged by
default, including when tools declare arrays or other non-string parameters.
To opt into schema-aware recovery, send the request extension
`"tool_argument_recovery": "buffered"` (with the OpenAI Python client, use
`extra_body={"tool_argument_recovery": "buffered"}`). The default is `"none"`;
other values are rejected by request validation.

Buffered mode emits each call's ID/name when established and continues emitting
independent content/reasoning. Only argument bytes wait until the entire
generation finishes, including for already-valid arguments. This does not
provide early per-call completion. Complete arguments can then recover, for
example, a JSON-encoded array string into a schema-declared array. Already-correct
strings are preserved; actual objects/lists targeting strings use compact UTF-8
JSON without sorting keys. Nonfinite or incompatible values are not repaired.
This option does not enforce the full tool schema or change non-streaming
recovery behavior.

## Supported Parsers

Use `--tool-call-parser` to select a parser for your model family:

| Parser | Aliases | Models | Format |
|--------|---------|--------|--------|
| `auto` | | Any model | Auto-detects format (tries all parsers) |
| `mistral` | | Mistral, Devstral | `[TOOL_CALLS]` JSON array |
| `qwen` | `qwen3` | Qwen, Qwen3 | `<tool_call>` XML or `[Calling tool:]` |
| `qwen3_xml` | `qwen3.5`, `qwen3_coder` | Qwen 3.5, Qwen3 Coder | Typed streaming XML |
| `llama` | `llama3`, `llama4` | Llama 3.x, 4.x | `<\|python_tag\|>` JSON, bare JSON, or `<function=name>` tags |
| `hermes` | `nous` | Hermes, NousResearch | `<tool_call>` JSON in XML |
| `deepseek` | `deepseek_v3`, `deepseek_r1` | DeepSeek V3, R1 | Unicode delimiters |
| `deepseek_v4` | `dsml` | DeepSeek V4 Flash / Flash-0731 | DSML typed parameters |
| `gemma4` | | Gemma 4 | `<\|tool_call>` call blocks |
| `kimi` | `kimi_k2`, `moonshot` | Kimi K2, Moonshot | `<\|tool_call_begin\|>` tokens |
| `granite` | `granite3` | IBM Granite 3.x, 4.x | `<\|tool_call\|>` or `<tool_call>` |
| `nemotron` | `nemotron3` | NVIDIA Nemotron | `<tool_call><function=...><parameter=...>` |
| `minimax` | `minimax_m2` | MiniMax M2 | `<minimax:tool_call>` XML |
| `harmony` | `gpt-oss` | GPT-OSS | Harmony commentary control tokens |
| `poolside_v1` | | Poolside v1 / Laguna | Typed `<arg_key>`/`<arg_value>` XML |
| `step3p5` | `step` | Step 3.5 | `<tool_call><function=...>` XML |
| `xlam` | | Salesforce xLAM | JSON with `tool_calls` array |
| `functionary` | `meetkai` | MeetKai Functionary | Multiple function blocks |
| `glm47` | `glm4` | GLM-4.7, GLM-4.7-Flash | `<tool_call>` with `<arg_key>`/`<arg_value>` XML |
| `lfm2` | `lfm2.5` | Liquid LFM2, LFM2.5 | `<\|tool_call_start\|>[fn(arg=value)]<\|tool_call_end\|>` Python-call list |
| `minicpm` | `minicpm5` | MiniCPM5 | `<function name=".."><param name="..">..</param></function>` XML |
| `phi4_mini_json` | `phi4_mini`, `phi4` | Microsoft Phi-4-mini | `functools[{"name": .., "arguments": {..}}]` JSON list; needs `--chat-template` |
| `pythonic` | `gemma3_pythonic` | Llama 3.2/4, ToolACE, Gemma 3 | Bare Python list of keyword-only calls `[fn(arg=value), ...]`, no marker; needs a template that shows the tools (`gemma3_pythonic` goes with the shipped Gemma 3 one) |

## Model Examples

### Mistral / Devstral

```bash
# Devstral Small (optimized for coding and tool use)
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Mistral Instruct
vllm-mlx serve mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

**Tokenizer (tekken checkpoints).** Checkpoints that ship `tekken.json` (Ministral 3,
Devstral Small 2, Mistral Small 3.x, Nemo) are served with their `tokenizer.json` fast
tokenizer, checked at load time to be identical to the mistral-common one (every token id
maps to the same bytes, same control-token names, same BOS/EOS, same encoding of a battery
of probe texts), so `[INST]`, `[TOOL_CALLS]` and the other control tokens of the rendered
prompt are encoded as the control tokens the model was trained on. Without a
`tokenizer.json`, or if it differs, the server logs a warning and these models are
prompted with the control tokens spelled out as plain text.

**Chat roles (tekken checkpoints).** The prompt is rendered by the model's own
`chat_template.jinja`, which is stricter than mistral-common's renderer. Requests the
template rejects fail with the template's message:

| Messages | Message |
|----------|---------|
| a second `system` message, or `system` after the first message | `Only user, assistant and tool roles are supported, got system.` |
| `user, user` or `assistant, assistant` (without tool calls) in a row | `After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.` |

mistral-common itself accepts both (it emits the extra system prompt in place, or joins
consecutive same-role messages with a blank line). Merge such messages in the client; the
server does not rewrite them.

### Qwen

```bash
# Qwen3
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser qwen
```

### Llama

```bash
# Llama 3.2
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser llama
```

### DeepSeek

```bash
# DeepSeek V3
vllm-mlx serve mlx-community/DeepSeek-V3-0324-4bit \
  --enable-auto-tool-choice --tool-call-parser deepseek
```

DeepSeek V4 uses a different wire format and must use both V4 parsers. The
programmatic encoder supports the published `DeepSeek-V4-Flash` preview and
the `DeepSeek-V4-Flash-0731` profile:

```bash
vllm-mlx serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4
```

The preview profile accepts `high` and `max`; the 0731 profile accepts `low`,
`high`, and `max`. OpenAI `reasoning_effort` values are normalized as follows:
unspecified → `high`, `minimal`/`low`/`medium` → `low`, `high`/`xhigh` →
`high`, and `max` → `max`. On the preview checkpoint, `low` normalizes to
`high`. Streaming buffers partial DSML markers and emits exactly one structured
tool call after the block is complete; truncated markup is returned as text.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    messages=[{"role": "user", "content": "What is the weather in Prague?"}],
    reasoning_effort="high",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(response.choices[0].message.tool_calls[0].function.arguments)
```

### IBM Granite

```bash
# Granite 4.0
vllm-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite
```

### NVIDIA Nemotron

```bash
# Nemotron 3 Nano
vllm-mlx serve mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit \
  --enable-auto-tool-choice --tool-call-parser nemotron
```

### GLM-4.7

```bash
# GLM-4.7 Flash
vllm-mlx serve lmstudio-community/GLM-4.7-Flash-MLX-8bit \
  --enable-auto-tool-choice --tool-call-parser glm47
```

### Kimi K2

```bash
# Kimi K2
vllm-mlx serve mlx-community/Kimi-K2-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser kimi
```

### Microsoft Phi-4-mini

Phi-4-mini's own chat template ignores the OpenAI `tools=` argument: the model is
never told which tools exist and answers in prose. The package ships vLLM's tool
template (adapted, see the header of the file) and `--chat-template` installs it:

```bash
TEMPLATE=$(python -c 'import vllm_mlx.templates as t; print(t.PHI4_MINI_TOOL_TEMPLATE)')
vllm-mlx serve mlx-community/Phi-4-mini-instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser phi4_mini_json \
  --chat-template "$TEMPLATE"
```

`--chat-template` takes a file path or a one-line inline template (`\n`, `\t`, `\r`
are decoded, everything else is kept as typed), applies to every engine (Simple,
Batched, text and multimodal), and is refused with `--models-config`. It is
server-wide, so the shipped template renders requests **without tools** byte-for-byte
like the model's own template (tests pin this); only requests with tools get the
function-calling system prompt.
The model answers a call with `functools[{"name": ..., "arguments": {...}}]`; the
parser finds the end of the list with a bracket scan, so arguments containing `]`
are safe, and a list it cannot fully parse stays in `content`. It streams: each call
is emitted once, when its list closes.

### Pythonic tool calls (Gemma 3, Llama 3.2/4, ToolACE)

These models answer a tool request with nothing but a Python list of keyword-only calls,
no marker tokens: `[get_weather(city='Paris'), get_time(tz='UTC')]`. The `pythonic`
parser (a port of vLLM's, same accepted language) reads it:

* the answer is a tool call only when its **first character is `[`**; anything else
  (including leading whitespace) is plain content, streaming or not;
* every element must be a call with keyword arguments only and literal values (strings,
  numbers, `True`/`False`/`None` or `true`/`false`/`null`, lists, tuples/sets as lists,
  dicts with literal keys); a positional argument, `**kwargs`, a variable, an
  expression, an empty list or any other element leaves the **whole** answer in
  `content`; dotted names (`a.b(x=1)`) are kept dotted;
* text after the closing `]` is returned as `content` next to the calls;
* streaming withholds a leading `[...` until it is known to be a call list or not: the
  calls of a list are emitted once, with their absolute index, when its `]` arrives
  (all or nothing, so streaming returns what non-streaming returns), and bracket text
  that is not a call list, or a list the model never closed, is released as content.

Gemma 3's own template never shows the model the tools, so the package ships vLLM's
Gemma 3 pythonic template (adapted, see the header of the file; the parser alias
`gemma3_pythonic` is the same parser):

```bash
TEMPLATE=$(python -c 'import vllm_mlx.templates as t; print(t.GEMMA3_PYTHONIC_TOOL_TEMPLATE)')
vllm-mlx serve mlx-community/gemma-3-4b-it-qat-4bit --continuous-batching \
  --enable-auto-tool-choice --tool-call-parser pythonic \
  --chat-template "$TEMPLATE"
```

Requests **without tools** render byte-for-byte like Gemma's own template (the `<bos>`
included, which every engine path handles: SimpleEngine and mlx-lm do not add a second
one, the multimodal batching path tokenizes without special tokens). With tools the
first user turn carries vLLM's function-calling prompt; assistant `tool_calls` are
replayed as the list the model is asked to write and `tool` results become
`<tool_response>` user turns. For other model families (Llama 3.2/4) use
`--tool-call-parser pythonic` with that family's own pythonic template (vLLM ships
`tool_chat_template_llama3.2_pythonic.jinja` and `tool_chat_template_llama4_pythonic.jinja`);
they are not bundled here.

Model quality is the limit, not the parser: `gemma-3-4b-it-qat-4bit` writes correct
calls for simple tools (`get_weather(city="Paris")`, two parallel calls, two different
tools) but wraps the call in quotes for some schemas (`['echo_args(text=...)']`, a list
holding a string, which is content) and often ends its turn at once after a tool result
(measured with mlx-lm at temperature 0 on the same prompt, so not the server's doing).

### Salesforce xLAM

```bash
# xLAM
vllm-mlx serve mlx-community/xLAM-2-fc-r-4bit \
  --enable-auto-tool-choice --tool-call-parser xlam
```

## Auto Parser

If you're not sure which parser to use, the `auto` parser tries to detect the format automatically:

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser auto
```

The auto parser tries formats in this order:
1. Gemma 4 (`<|tool_call>call:name...<tool_call|>`)
2. DeepSeek V4 DSML (`<｜DSML｜tool_calls>`)
3. Mistral (`[TOOL_CALLS]`)
4. Qwen bracket (`[Calling tool:]`)
5. Nemotron (`<tool_call><function=...><parameter=...>`)
6. Qwen/Hermes XML (`<tool_call>{...}</tool_call>`)
7. Llama (`<function=name>{...}</function>`)
8. Raw JSON

## Streaming Tool Calls

Tool calls work with streaming. The tool call information is sent when the model finishes generating:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's 25 * 17?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
```

## Handling Tool Results

After receiving a tool call, execute the function and send the result back:

```python
import json

# First request - model decides to call a tool
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# Get the tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Execute the function (your implementation)
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# Send result back to model
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)}
    ],
    tools=[weather_tool]
)

print(response.choices[0].message.content)
# "The weather in Tokyo is sunny with a temperature of 22C."
```

## Think Tag Handling

Models that produce `<think>...</think>` reasoning tags (like DeepSeek-R1, Qwen3, GLM-4.7) are handled automatically. The parser strips thinking content before extracting tool calls, so reasoning tags never interfere with tool call parsing.

This works even when `<think>` was injected in the prompt (implicit think tags with only a closing `</think>`).

Tool calls are parsed from the content part only. With `--reasoning-parser` the reasoning is separated first and never reaches the tool parser; call-shaped markup the model drafts while thinking stays in `reasoning`, and a call that exists only inside the thinking is not a call. See [Reasoning and Tool Calls](reasoning.md#reasoning-and-tool-calls).

## CLI Reference

| Option | Description |
|--------|-------------|
| `--enable-auto-tool-choice` | Enable automatic tool calling |
| `--tool-call-parser` | Select parser (see table above) |

See [CLI Reference](../reference/cli.md) for all options.
