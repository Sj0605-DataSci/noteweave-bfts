# noteweave-bfts

Best-First Tree Search (BFTS) for automated ML experiments. Used by the [NoteWeave](https://github.com/Sj0605-DataSci/noteweave-cli) extension and CLI as the coding agent's experiment engine.

## Install

```bash
pip install noteweave-bfts
```

## Usage

### As a NoteWeave tool (default)

When launched by the NoteWeave CLI or VS Code extension, BFTS runs in **NoteWeave mode**: LLM calls route through the NoteWeave backend server, and the process communicates via a JSON-lines control protocol on stdin/stdout.

```bash
python -m noteweave_bfts --config bfts_config.yaml --mode noteweave
```

### Standalone

For testing without the NoteWeave server — uses direct OpenAI/Anthropic/Ollama calls:

```bash
python -m noteweave_bfts --config bfts_config.yaml --mode standalone
```

## 4-Stage Pipeline

1. **Initial implementation** -- get working baseline code
2. **Baseline tuning** -- optimize hyperparameters
3. **Creative research** -- try novel improvements
4. **Ablation studies** -- analyze what works and why

## Control Protocol

When running in NoteWeave mode, the process communicates via JSON lines:

**BFTS -> Runner (stdout):**
- `{"type": "status", "stage": "stage_1", "node_id": 3, "metric": 0.83}`
- `{"type": "llm_request", "req_id": "abc", "system": "...", "user": "..."}`
- `{"type": "done", "success": true, "summary": {...}}`

**Runner -> BFTS (stdin):**
- `{"type": "llm_response", "req_id": "abc", "content": "..."}`
- `{"type": "command", "action": "stop"}`
- `{"type": "command", "action": "query", "target": "journal"}`

## License

MIT
