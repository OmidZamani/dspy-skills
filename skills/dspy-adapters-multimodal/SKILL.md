---
name: dspy-adapters-multimodal
version: "2.0.0"
dspy-compatibility: "3.2.1"
description: This skill should be used when the user asks to "choose a DSPy adapter", "use JSONAdapter", "use XMLAdapter", "enable native function calling", "send images, audio, or files to DSPy", mentions `dspy.ChatAdapter`, `dspy.JSONAdapter`, `dspy.XMLAdapter`, `dspy.Image`, `dspy.Audio`, `dspy.File`, structured outputs, or multimodal DSPy signatures.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy Adapters and Multimodal I/O

## Goal

Choose an adapter deliberately and model image, audio, and file inputs with DSPy's typed primitives.

## Adapter Selection

| Adapter | Use it for |
|---------|------------|
| `dspy.ChatAdapter()` | Default, human-readable field markers, broad model compatibility |
| `dspy.JSONAdapter()` | Structured JSON output and native function calling where supported |
| `dspy.XMLAdapter()` | XML-tagged fields when XML is easier for the target LM to follow |
| `dspy.TwoStepAdapter()` | A separate extraction pass when parsing needs extra help |

Configure globally or for a limited scope:

```python
import dspy

dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.JSONAdapter(),
)

with dspy.context(adapter=dspy.XMLAdapter()):
    result = dspy.Predict("question -> answer")(question="What is DSPy?")
```

## Native Function Calling

`JSONAdapter` enables native function calling by default. `ChatAdapter` keeps text parsing by default. Override either behavior explicitly:

```python
chat_native = dspy.ChatAdapter(use_native_function_calling=True)
json_manual = dspy.JSONAdapter(use_native_function_calling=False)
```

DSPy falls back to manual parsing when the configured LM does not support native function calling.

## Image Inputs

```python
class DescribeImage(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    description: str = dspy.OutputField()

describe = dspy.Predict(DescribeImage)
result = describe(image=dspy.Image("./diagram.png"))
```

Pass a local path, HTTP URL, bytes, PIL image, or existing data URI directly to `dspy.Image(...)`.

## Audio and File Inputs

```python
class SummarizeAudio(dspy.Signature):
    audio: dspy.Audio = dspy.InputField()
    summary: str = dspy.OutputField()

audio = dspy.Audio.from_file("./meeting.wav")
summary = dspy.Predict(SummarizeAudio)(audio=audio)
```

```python
class SummarizeFile(dspy.Signature):
    file: dspy.File = dspy.InputField()
    summary: str = dspy.OutputField()

document = dspy.File.from_path("./research.pdf")
summary = dspy.Predict(SummarizeFile)(file=document)
```

Provider capabilities vary. Verify that the selected model accepts the media type before deployment.

## Best Practices

1. Start with `ChatAdapter`; switch only for a measured reason.
2. Use typed signatures for structured output.
3. Test adapter behavior against the exact production model.
4. Avoid deprecated `Image.from_file()` and `Image.from_url()` helpers; call `dspy.Image(...)`.
5. Keep local file handling and uploaded file IDs within provider policy.

## Related Skills

- Design signatures: [dspy-signature-designer](../dspy-signature-designer/SKILL.md)
- Build tool agents: [dspy-react-agent-builder](../dspy-react-agent-builder/SKILL.md)

## Official Documentation

- **Adapters guide**: https://dspy.ai/learn/programming/adapters/
- **Tools guide**: https://dspy.ai/learn/programming/tools/
- **XMLAdapter API**: https://dspy.ai/api/adapters/XMLAdapter/
- **Image API**: https://dspy.ai/api/primitives/Image/
- **Audio API**: https://dspy.ai/api/primitives/Audio/
