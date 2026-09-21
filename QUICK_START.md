# QUICK START — Running one participant session (e.g. P001)

A full single-participant session from start to finish. Copy-paste in order.

---

## 0. Pre-flight check (once, before the session)

```bash
# Activate the virtual environment
.venv\Scripts\activate

# Check .env (both keys required)
grep -E "ELEVENLABS_API_KEY|GEMINI_API_KEY" .env

# Ollama daemon + embeddinggemma model
ollama list | grep embeddinggemma   # if missing: ollama pull embeddinggemma

# Confirm the base model file exists
ls model_weights/gemma-3-4b-it-q4_0.gguf
```

---

## 1. Phase 1 — Personal-info collection (~5 min)

Run the server on the same PC as Unity:

```bash
python server.py -p P001 --phase 1
```

- In Unity (right-hand controller): **A** = hear the question, **Trigger** (hold) = record the
  answer, **joystick** ◀▶ = prev/next, **hold B** = finish.
- After you finish (hold B) with all 10 answered, the closing line plays → the server auto-shuts
  down ~8 s later (`PHASE1_SHUTDOWN_DELAY_SEC`).
- Output: `participants/P001/phase1/conversation_log.json` + `answers.json`.
- Server output is written to `participants/P001/server_phase1_<timestamp>.log`
  (pass `--debug` to echo to the console instead).

Troubleshooting: if the server won't start, check for a port 8000 conflict
(`netstat -ano | findstr 8000`).

---

## 2. Phase 2 — Memory build (~10–15 min, automated)

```bash
python build_memory.py -p P001
```

The console shows only a start line and the final status; detailed progress goes to
`logs/phase2.log`. Stages:
1. Load Phase 1 → generate Gemini scenarios (10 × 20 turns, ~2 min)
2. `[Phase2] [1/3] done` — in_context (instant, file I/O only)
3. `[Phase2] [2/3] done` — retrieval FAISS (~30 s)
4. `[Phase2] [3/3] done` — parametric LoRA (~8–10 min)

Completion markers:
- `[Phase2] P001 all memory build complete` (and `[build_memory] final phase2_status = done`)
- `participants/P001/build_cost.json` created
- `model_weights/P001_lora_persona.gguf` created (~57 MB)

On failure:
```bash
cat participants/P001/status.json   # check phase2_error
tail -50 logs/phase2.log            # full build + LoRA training log
```

---

## 3. Phase 3 — Free dialog across the three conditions (10–20 min each)

### 3-1. Check condition_order

```bash
cat participants/P001/status.json | python -c "import sys, json; print(json.load(sys.stdin)['condition_order'])"
# e.g.: ['in_context', 'retrieval', 'parametric']
```

### 3-2. Run the server per condition (one at a time)

```bash
# First condition (order[0])
python server.py -p P001 --phase 3 -c 0
# The server log shows the resolved condition, e.g. [server] -c 0 → condition='in_context'
# Free dialog in Unity → stop with Ctrl+C, or trigger /phase3/abort from Unity
```

A short participant break before the next condition is recommended.

```bash
# Second condition
python server.py -p P001 --phase 3 -c 1

# Third condition
python server.py -p P001 --phase 3 -c 2
```

Per-session logs: `participants/P001/phase3/<cond>_log.json` (turn-by-turn telemetry) and
`participants/P001/server_phase3_<timestamp>.log` (console output; `--debug` to echo live).

---

## 4. Analysis (after the session)

### 4-1. CLI verification without Unity (optional)

```bash
# factual 10 prompts — memory-accuracy test
python tools/test_conditions_cli.py -p P001

# natural 7 turns — proactive-recall test in natural conversation
python tools/test_conditions_cli.py -p P001 --prompts natural
```

Results: `participants/P001/cli_test_results{_natural}.json`

### 4-2. Combined cost report

```bash
python tools/cost_report.py -p P001
```

Output: per-condition build wall-clock + Gemini scenario-generation cost + Phase 3 runtime
tokens / latency (llm_ms).

---

## 5. Reset (when needed)

```bash
# Full reset (start over from Phase 1)
python reset.py -p P001 --all --yes

# A single phase
python reset.py -p P001 --phase 1 --yes
python reset.py -p P001 --phase 2 --yes
python reset.py -p P001 --phase 3 --yes

# Only one condition's Phase 3 (to re-run that condition)
python reset.py -p P001 --phase 3 --condition in_context --yes
```

---

## Checklist (per participant)

- [ ] Phase 1: all 10 questions answered → every entry in `answers.json` has `revisions`
- [ ] Phase 2: `status.json::phase2_status = done`
- [ ] `model_weights/P001_lora_persona.gguf` exists (50 MB+)
- [ ] Phase 3: all three conditions run → each `phase3/<cond>_log.json` accumulated
- [ ] `build_cost.json` created
- [ ] Participant questionnaire completed (separate)

---

## Known issues / quick fixes

| Symptom | Cause | Fix |
|---|---|---|
| Phase 2: `GEMINI_API_KEY missing` | `.env` not set | add `GEMINI_API_KEY` to `.env` |
| Phase 3: server fails to load the model | out of VRAM / another process holding it | close browsers/games and restart |
| parametric hallucinates a name (e.g. "Kim Minji") | stale LoRA left over | re-build after `reset.py --phase 2` |
| Unity gets 401/403 | STT key expired | refresh `.env::ELEVENLABS_API_KEY` |
| `ollama` command not found | not on PATH / Ollama not installed | install from https://ollama.ai |
| llama-cpp-python loaded as a CPU build | default pip wheel | reinstall the CUDA build (see README Installation) |
| Phase 1 voice cache plays an old question | question YAML changed but cache remained | delete `voice/voice_cache/<voice_id>/` and restart |

---

## See also

- **Full setup guide**: `README.md`
- **VR client setup**: `RISNSFP_VR_client_setup.md`
