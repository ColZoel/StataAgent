# StataAgent

> **Beta v0.0.1** — experimental; interfaces and behavior may change without notice.

StataAgent is an AI-powered agent for interactive data exploration and analysis directly within Stata. Built on [pydantic-ai](https://github.com/pydantic/pydantic-ai) and LiteLLM, it interprets natural-language questions and Stata commands, converts them into executable Stata code, and returns results in your terminal.

---

## Compatibility

| Component | Requirement |
|-----------|-------------|
| **Stata** | **17 or newer** (requires `pystata`, which ships with Stata 17+) |
| **Python** | 3.10 or newer |
| **pydantic-ai** | ≥ 1.87.0 |
| **pydantic-ai-litellm** | latest (installed alongside pydantic-ai) |

> **Stata 16 and earlier are not supported.** `pystata` — the Python API used to execute Stata commands — was introduced in Stata 17 and is not available for earlier versions.

---

## Model / Provider Caution

> **Not all inference providers work reliably with this tool.**
>
> StataAgent requires a model that supports structured tool-calling (function calling). Many free-tier and serverless inference endpoints either do not implement the tool-calling protocol correctly or apply aggressive rate limits that break multi-step agentic loops.
>
> **The only provider found to be consistently usable so far is [Nscale](https://nscale.com) via an OpenAI-compatible endpoint.** Other providers (HuggingFace Serverless, SambaNova free tier, etc.) have caused tool-call parsing failures or silent response truncation during testing.
>
> If you use a different provider, confirm it supports OpenAI-style function calling and has sufficient token limits (≥ 4 096 output tokens). Commercial providers like Together AI, Fireworks, and Groq are more likely to work than free-tier serverless endpoints, but have not been systematically tested.

### Recommended config (Nscale / OpenAI-compatible)

```yaml
# config.yaml
provider: openai_compatible
model: <nscale-model-id>          # e.g. meta-llama/Llama-3.3-70B-Instruct
base_url: https://inference.nscale.com/v1
api_key_env: NSCALE_API_KEY
temperature: 0.3
max_tokens: 4096
commentary: false
```

```bash
export NSCALE_API_KEY=<your-key>
```

---

## Features

- **Natural language queries** — ask questions in plain English; StataAgent maps them to Stata commands.
- **Structured tools** — dedicated wrappers for `summarize`, `tabulate`, `regress`, and `describe` produce clean, predictable output.
- **Escape hatch** — `run_stata` executes arbitrary Stata code for commands not covered by the structured tools (`xtreg`, `margins`, `xtset`, etc.).
- **Metadata awareness** — variable names and labels are loaded at startup so the model can reference them without hallucinating names.
- **Multi-turn history** — conversation context is preserved across queries within a session.
- **Commentary toggle** — `--commentary` / `--no-commentary` controls whether the model adds plain-language interpretation after raw Stata output.

---

## Installation

### Prerequisites

- Stata 17+ (with `pystata` available — it ships with the Stata installation)
- Python 3.10+
- An API key for your chosen inference provider

### Setup

```bash
git clone https://github.com/ColZoel/StataAgent.git
cd StataAgent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy and edit the config:

```bash
cp config.yaml my_config.yaml   # optional — edit in place if you prefer
```

Set your API key:

```bash
export NSCALE_API_KEY=<your-key>   # or whichever variable api_key_env points to
```

---

## Usage

### Web UI

```bash
python web.py                 # serves http://127.0.0.1:8765 and opens a browser tab
# or equivalently:
python main.py --ui           # same UI, launched through the main entry point
```

The UI mirrors the Stata desktop layout — Results and Command windows in the
center, a Review (history) pane on the left, and Variables / Graphs /
Properties on the right. Stata output streams into the Results window live as
each command executes, and any graphs written to the working directory are
picked up automatically.

Drag and drop files anywhere in the window, Stata-GUI style: a `.dta` file is
loaded into Stata immediately (the `use` log appears in the Results window and
the Variables pane updates), and a `.do` file is attached as context for the
agent's next turn — ask "explain this do-file" or "run this".

### CLI

```bash
python main.py
# or, with overrides:
python hf_agent.py --config my_config.yaml
python hf_agent.py --provider openai_compatible --base-url https://inference.nscale.com/v1 --model meta-llama/Llama-3.3-70B-Instruct
python hf_agent.py --no-commentary
python hf_agent.py --temperature 0.1 --max-tokens 2048
```

Once running, type queries at the `>` prompt. Type `quit` or `exit` to end the session.

---

## Examples

### Load a dataset and describe it

```
> load /path/to/data.dta and describe it
```

StataAgent calls `load_data` then `describe_dataset` and returns the variable list with types and labels.

### Descriptive statistics

```
> Summarize the wage and education variables
```

Internally executes:

```stata
summarize wage education
```

### Regression

```
> What is the effect of education on wages, controlling for experience and gender?
```

Internally executes:

```stata
regress wage education experience gender
```

### Cross-tabulation

```
> Show the distribution of homeownership by year
```

Internally executes:

```stata
tabulate homeown year
```

### Arbitrary Stata command

```
> Run a fixed-effects regression of wage on education, with individual fixed effects (id) and year fixed effects
```

Internally executes (via `run_stata`):

```stata
xtset id year
xtreg wage education i.year, fe robust
```

### CLI flag examples

```bash
# Start with commentary disabled (raw Stata output only)
python hf_agent.py --no-commentary

# Override model without editing config.yaml
python hf_agent.py --provider openai_compatible \
  --base-url https://inference.nscale.com/v1 \
  --model meta-llama/Llama-3.3-70B-Instruct

# Reset saved Stata installation config and re-run discovery
python hf_agent.py --reset
```

---

## Notes

- Each session re-loads the dataset from disk when `load_data` is called. There is no in-memory persistence across sessions, so StataAgent is not recommended for datasets above roughly 100 000 observations.
- `pystata` must be importable from your Python environment. If it is not found, verify that your `PYTHONPATH` includes the `utilities/python` directory inside your Stata installation.

---

## Contributing

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push and open a Pull Request.

---

## Acknowledgments

- [pydantic-ai](https://github.com/pydantic/pydantic-ai) — typed AI agent framework
- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM provider backend
- StataCorp for `pystata` and the Stata platform
