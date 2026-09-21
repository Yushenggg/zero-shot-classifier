# zero-shot

Inspired by [jev.ai](https://jev.ai) and its ability to turn an arbitrary "state"
plus a set of choices into a decision with real probabilities, no training
required.

## How it works (roughly)

1. Build a prompt from the state (arbitrary JSON context), the task instructions,
   and the choice criteria, ending right at the decision point.
2. Run a forward pass with a local Gemma model and read the next-token
   distribution from the full-vocabulary logits.
3. For each option, score its exact continuation token-by-token, plus the EOS
   token, giving `log P(option)`.
4. Softmax those scores into probabilities and pick the highest.

Including EOS is what lets a longer but more complete answer win over a short one.

## Usage

```bash
uv sync
uv run zero-shot -q examples/response_question.json -s examples/response_state.json
uv run zero-shot-serve   # web UI at http://127.0.0.1:8000
```

Model name and local save directory live in `config.toml`. See the CLI
(`uv run zero-shot --help`) for options.
