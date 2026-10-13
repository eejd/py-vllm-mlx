# Reasoning Models

vllm-mlx supports reasoning models that show their thinking process before giving an answer. Models like Qwen3 and DeepSeek-R1 wrap their reasoning in `<think>...</think>` tags, and vllm-mlx can parse these tags to separate the reasoning from the final response.

## Why Use Reasoning Parsing?

When a reasoning model generates output, it typically looks like this:

```
<think>
Let me analyze this step by step.
First, I need to consider the constraints.
The answer should be a prime number less than 10.
Checking: 2, 3, 5, 7 are all prime and less than 10.
</think>
The prime numbers less than 10 are: 2, 3, 5, 7.
```

Without reasoning parsing, you get the raw output with the tags included. With reasoning parsing enabled, the thinking process and final answer are separated into distinct fields in the API response.

## Getting Started

### Start the Server with Reasoning Parser

```bash
# For Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# For DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### API Response Format

When reasoning parsing is enabled, the API response includes a `reasoning` field:

**Non-streaming response:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The prime numbers less than 10 are: 2, 3, 5, 7.",
      "reasoning": "Let me analyze this step by step.\nFirst, I need to consider the constraints.\nThe answer should be a prime number less than 10.\nChecking: 2, 3, 5, 7 are all prime and less than 10."
    }
  }]
}
```

**Streaming response:**

Chunks are sent separately for reasoning and content. During the reasoning phase, chunks have `reasoning` populated. When the model transitions to the final answer, chunks have `content` populated:

```json
{"delta": {"reasoning": "Let me analyze"}}
{"delta": {"reasoning": " this step by step."}}
{"delta": {"reasoning": "\nFirst, I need to"}}
...
{"delta": {"content": "The prime"}}
{"delta": {"content": " numbers less than 10"}}
{"delta": {"content": " are: 2, 3, 5, 7."}}
```

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What are the prime numbers less than 10?"}]
)

message = response.choices[0].message
print("Reasoning:", message.reasoning)  # The thinking process
print("Answer:", message.content)        # The final answer
```

### Streaming with Reasoning

```python
reasoning_text = ""
content_text = ""

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 2 + 2 = ?"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning') and delta.reasoning:
        reasoning_text += delta.reasoning
        print(f"[Thinking] {delta.reasoning}", end="")
    if delta.content:
        content_text += delta.content
        print(delta.content, end="")

print(f"\n\nFinal reasoning: {reasoning_text}")
print(f"Final answer: {content_text}")
```

## Supported Parsers

### Qwen3 Parser (`qwen3`)

For Qwen3 models that use explicit `<think>` and `</think>` tags.

- Requires **both** opening and closing tags
- If tags are missing, output is treated as regular content
- Best for: Qwen3-0.6B, Qwen3-4B, Qwen3-8B and similar models

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

### DeepSeek-R1 Parser (`deepseek_r1`)

For DeepSeek-R1 models that may omit the opening `<think>` tag.

- More lenient than Qwen3 parser
- Handles cases where `<think>` is implicit
- Content before `</think>` is treated as reasoning even without `<think>`

```bash
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### DeepSeek-V4 Parser (`deepseek_v4`)

DeepSeek V4 Flash can end an implicit reasoning block by opening a DSML tool
call, even when `</think>` is absent. Use the V4 reasoning parser together with
the V4 tool parser so the DSML tail is routed to structured tool-call output:

```bash
vllm-mlx serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4
```

The encoder detects the published preview and 0731 reasoning-effort profiles.
For OpenAI requests, unspecified effort selects `high`; `minimal`, `low`, and
`medium` select `low`; `high` and `xhigh` select `high`; and `max` selects
`max`. The preview profile normalizes `low` to its `high` behavior.

## Reasoning and Tool Calls

When a reasoning parser and a tool parser are both configured, the reasoning is separated first and the tool parser is handed the **content part only**. This holds on every endpoint (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`), streaming or not:

- Tool-call markup inside the thinking stays in `reasoning`, untouched. A model that drafts its call while thinking (`Let me make the call: <tool_call>...`) and then makes it after `</think>` yields one call, not two.
- A call that exists only inside the thinking is not a call. This is the same answer as without a reasoning parser, where the tool parsers strip the thinking themselves, and as vLLM gives. The server logs a WARNING once per request when the reasoning holds tool-call markup and the response has no call (sizes only, never the text), so an agent loop that stalls on a model that writes its call inside `<think>` leaves a trace.
- A generation cut off inside the thinking (`finish_reason: length`, `</think>` never written) has no content and therefore no tool calls. When the chat template opens `<think>` in the prompt (Qwen3.5/3.8) the non-streaming endpoints read an output with no tag at all as exactly that, as the streaming path does.

Gemma 4 follows the same rule: a `<|tool_call>` inside an unclosed thought channel is reasoning, not a call.

GPT-OSS (`--reasoning-parser gpt_oss` or `harmony` with `--tool-call-parser harmony` / `gpt-oss`) is the one exception to "content part only": its calls live in the Harmony commentary channel, which a reasoning parser keeps out of its content. The harmony tool parser reads the channels itself (the analysis channel is never content or a call), so it is handed the raw output, control tokens intact, and the reasoning parser supplies the `reasoning`. A call before a final message, a call that ends the generation (the closing `<|call|>` is a stop token and is usually missing from the text) and a call mentioned only in the analysis (not a call) all behave as above, on every endpoint, streaming or not, for deltas of any size.

## How It Works

The reasoning parser uses text-based detection to identify thinking tags in the model output. During streaming, it tracks the current position in the output to correctly route each token to either `reasoning` or `content`.

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │     reasoning       ││       content       │
                     └─────────────────────┘└─────────────────────┘
```

The parsing is stateless and uses the accumulated text to determine context, making it robust for streaming scenarios where tokens may arrive in arbitrary chunks.

## Tips for Best Results

### Prompting

Reasoning models work best when you encourage step-by-step thinking:

```python
messages = [
    {"role": "system", "content": "Think through problems step by step before answering."},
    {"role": "user", "content": "What is 17 × 23?"}
]
```

### Handling Missing Reasoning

Some prompts may not trigger reasoning. In these cases, `reasoning` will be `None` and all output goes to `content`:

```python
message = response.choices[0].message
if message.reasoning:
    print(f"Model's thought process: {message.reasoning}")
print(f"Answer: {message.content}")
```

### Temperature and Reasoning

Lower temperatures tend to produce more consistent reasoning patterns:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    temperature=0.3  # More focused reasoning
)
```

## Backward Compatibility

When `--reasoning-parser` is not specified, the server behaves as before:
- Thinking tags are included in the `content` field
- No `reasoning` field is added to responses

This ensures existing applications continue to work without changes.

## Example: Math Problem Solver

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def solve_math(problem: str) -> dict:
    """Solve a math problem and return reasoning + answer."""
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a math tutor. Show your work."},
            {"role": "user", "content": problem}
        ],
        temperature=0.2
    )

    message = response.choices[0].message
    return {
        "problem": problem,
        "work": message.reasoning,
        "answer": message.content
    }

result = solve_math("If a train travels 120 km in 2 hours, what is its average speed?")
print(f"Problem: {result['problem']}")
print(f"\nWork shown:\n{result['work']}")
print(f"\nFinal answer: {result['answer']}")
```

## Curl Examples

### Non-streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}]
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}],
    "stream": true
  }'
```

## Troubleshooting

### No reasoning field in response

- Make sure you started the server with `--reasoning-parser`
- Check that the model actually uses thinking tags (not all prompts trigger reasoning)

### Reasoning appears in content

- The model may not be using the expected tag format
- Try a different parser (`qwen3` vs `deepseek_r1`)

### Truncated reasoning

- Increase `--max-tokens` if the model is hitting the token limit mid-thought

## Related

- [Supported Models](../reference/models.md) - Models that support reasoning
- [Server Configuration](server.md) - All server options
- [CLI Reference](../reference/cli.md) - Command line options
