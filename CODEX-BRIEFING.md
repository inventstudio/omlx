# CODEX-BRIEFING: oMLX Outlines Structured Output Integration

**Branch:** `feature/outlines-structured-output`
**Repo:** `inventstudio/omlx`
**PR-fähig gegen:** `jundot/omlx` (upstream)
**Erstellt von:** Claude (via David/Zora)
**Datum:** 19.03.2026

---

## Was wurde gebaut?

Token-Level JSON Schema Enforcement für oMLX über Outlines. Wenn ein Request mit `response_format: { type: "json_schema", json_schema: { schema: {...} } }` kommt UND Outlines installiert ist, wird jedes generierte Token gegen die Schema-State-Machine geprüft. Invalide Tokens werden auf `-inf` gesetzt. Das Modell kann nur noch valides JSON produzieren.

**Ohne Outlines:** Graceful Fallback auf die bestehende Prompt-Injection-Methode (kein Breaking Change).

---

## Geänderte Dateien (7 Files, 244 Zeilen)

### 1. `omlx/api/json_logits_processor.py` (NEU, 195 Zeilen)

Kern des Features. Enthält:

- **`OutlinesJSONLogitsProcessor`** — Wrapper-Klasse die Outlines' `JSONLogitsProcessor` an mlx-lm's `(tokens, logits) -> logits` Interface anpasst. Folgt exakt dem `ThinkingBudgetProcessor`-Pattern aus `omlx/api/thinking.py`.
- **`_get_or_create_outlines_processor()`** — Cache-Layer für kompilierte Prozessoren. Outlines kompiliert Schema → Regex → State Machine beim ersten Aufruf (1-5 Sek). Cache macht Folge-Requests instant. Max 32 Einträge, LRU-Eviction.
- **`extract_json_schema()`** — Extrahiert das Schema-Dict aus einem OpenAI-style `ResponseFormat` (Pydantic oder Dict).
- **`is_outlines_available()`** — Import-Check für graceful degradation.

**Potentielle Probleme die Codex prüfen muss:**
- Outlines' `JSONLogitsProcessor` erwartet normalerweise `torch.Tensor`. LM Studio löst das über einen Reshape-Wrapper (Zeile 126-138 im neuen Code). **Testen ob mx.array direkt funktioniert oder ob eine torch↔mx Konvertierung nötig ist.** Die Outlines MLX-Integration (`outlines.from_mlxlm`) könnte intern schon mx.array akzeptieren — aber der `JSONLogitsProcessor` aus `outlines.processors.structured` ist das low-level Interface und könnte torch erwarten.
- Falls torch-Kompatibilitätsproblem: Wrapper bauen mit `mx.array → numpy → torch.Tensor → JSONLogitsProcessor → numpy → mx.array`. Ist nicht ideal aber funktional.

### 2. `omlx/request.py` (1 Zeile)

```python
# Zeile 72-73 (nach thinking_budget):
json_schema: Optional[Dict[str, Any]] = None
```

Neues Feld in `SamplingParams`. `Dict` war bereits importiert.

### 3. `omlx/scheduler.py` (25 Zeilen, ab Zeile 1644)

Injection-Point in `_build_sampler_and_processors()`. Direkt nach dem `ThinkingBudgetProcessor`-Block:

```python
if sampling_params.json_schema is not None:
    try:
        from .api.json_logits_processor import OutlinesJSONLogitsProcessor, is_outlines_available
        if is_outlines_available():
            json_processor = OutlinesJSONLogitsProcessor(
                schema=sampling_params.json_schema,
                tokenizer=self.tokenizer,
            )
            logits_processors.append(json_processor)
    except Exception as e:
        logger.warning(f"Failed to create Outlines JSON processor: {e}")
```

**`self.tokenizer` ist verfügbar** — bestätigt durch Zora (Z.997 im Scheduler). Wird auch vom ThinkingBudgetProcessor genutzt (Z.1634).

### 4. `omlx/server.py` (14 Zeilen, 2 Stellen)

**OpenAI-Path** (nach Zeile 1835, nach thinking_budget):
```python
if response_format:
    from .api.json_logits_processor import extract_json_schema, is_outlines_available
    json_schema = extract_json_schema(response_format)
    if json_schema and is_outlines_available():
        chat_kwargs["json_schema"] = json_schema
```

**Anthropic-Path** (nach Zeile 3120, identisches Pattern):
Gleicher Code-Block.

**Wichtig:** Die bestehende Prompt-Injection (`_inject_json_instruction`) bleibt aktiv! Das ist beabsichtigt — Prompt-Guidance + Token-Enforcement zusammen sind besser als nur eines von beiden. Das Modell "will" schon JSON produzieren (durch Prompt) und wird zusätzlich "gezwungen" (durch Outlines).

### 5. `omlx/engine/batched.py` (2 Zeilen)

Zwei Stellen wo `SamplingParams` gebaut wird — jeweils:
```python
json_schema=kwargs.get("json_schema", None),
```
Hinzugefügt in `generate()` (Z.311) und `stream_generate()` (Z.377).

### 6. `omlx/engine/vlm.py` (2 Zeilen)

Identisch zu batched.py — `json_schema` in `generate()` (Z.627) und `stream_generate()` (Z.689) durchgereicht.

### 7. `pyproject.toml` (3 Zeilen)

Neue optional dependency group:
```toml
structured = [
    "outlines>=1.0.0",
]
```

Install: `pip install -e ".[structured]"`

---

## Setup-Anleitung für Mac Studio (64GB)

### Voraussetzungen

oMLX muss als editable install laufen (nicht Homebrew):

```bash
# 1. Homebrew-Installation stoppen
brew services stop omlx

# 2. Fork klonen
cd ~/Projects  # oder wo auch immer
git clone https://github.com/inventstudio/omlx.git omlx-dev
cd omlx-dev
git checkout feature/outlines-structured-output

# 3. Virtual Environment (oMLX braucht Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install mit structured dependency
pip install -e ".[structured]"

# 5. Outlines-Installation verifizieren
python -c "from outlines.processors.structured import JSONLogitsProcessor; print('OK')"

# 6. Server starten
omlx serve --model-dir ~/models --port 8200
```

### Test-Request

```bash
curl -s http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-27B-4bit",
    "messages": [
      {"role": "user", "content": "Extract the persons and locations from: David lives in Hamm and works with Dennis in Dortmund."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "entity_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "persons": {
              "type": "array",
              "items": {"type": "string"}
            },
            "locations": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["persons", "locations"],
          "additionalProperties": false
        }
      }
    },
    "temperature": 0.1,
    "max_tokens": 200
  }' | python3 -m json.tool
```

**Erwartetes Ergebnis:**
```json
{
  "persons": ["David", "Dennis"],
  "locations": ["Hamm", "Dortmund"]
}
```

---

## Bekannte Risiken & Debug-Hinweise

### Risiko 1: Outlines mx.array Kompatibilität (HÖCHSTE PRIORITÄT)

**Problem:** Outlines' `JSONLogitsProcessor` ist für `torch.Tensor` designed. In `omlx/api/json_logits_processor.py` Zeile 126-138 wird angenommen, dass es mit `mx.array` direkt klappt.

**Symptom:** `TypeError` oder `AttributeError` beim ersten Request mit json_schema.

**Debug:**
```python
# Test im Python REPL:
import mlx.core as mx
from outlines.processors.structured import JSONLogitsProcessor
from mlx_lm.utils import load

model, tokenizer = load("mlx-community/Qwen3.5-27B-4bit")
schema = '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}'
proc = JSONLogitsProcessor(schema, tokenizer)

# Simuliere einen Aufruf:
tokens = mx.array([1, 2, 3])
logits = mx.zeros((1, 152064))  # Qwen vocab size
result = proc(tokens, logits)  # <-- Crasht hier?
```

**Fix falls es crasht:** Konvertierung einbauen in `OutlinesJSONLogitsProcessor.__call__()`:

```python
import numpy as np

# mx -> numpy -> outlines -> numpy -> mx
tokens_np = np.array(tokens.tolist())
logits_np = np.array(logits.tolist(), dtype=np.float32)
logits_np = self._outlines_processor(tokens_np, logits_np)
return mx.array(logits_np)
```

### Risiko 2: Outlines first-compilation Timeout

**Problem:** Erste Kompilierung eines Schemas dauert 1-5 Sekunden. Bei Streaming-Requests könnte das Timeout verursachen.

**Symptom:** Erster Request mit neuem Schema ist langsam, danach instant (Cache greift).

**Fix:** Kein Fix nötig — Cache löst das nach dem ersten Call. Optional: Pre-warm Cache beim Serverstart für bekannte Schemas.

### Risiko 3: Continuous Batching + Mixed Requests

**Problem:** Batch enthält Requests MIT und OHNE json_schema. Der Logits-Processor wird pro Request in `_build_sampler_and_processors()` gebaut (Zeile 1601), aber der `BatchGenerator` arbeitet batch-weise.

**Debug:** Prüfen wie `_BoundarySnapshotBatchGenerator` mit per-request logits_processors umgeht. Zeile 2908: `logits_processors=[logits_processors]` — das ist eine Liste pro Request, also sollte es korrekt isoliert sein. Aber testen mit:
1. Request A: json_schema gesetzt
2. Request B: kein json_schema
3. Beide gleichzeitig senden

### Risiko 4: Tokenizer-Kompatibilität

**Problem:** Outlines braucht einen HuggingFace-kompatiblen Tokenizer. oMLX nutzt mlx-lm's Tokenizer, der ein `TokenizerWrapper` um einen HF-Tokenizer ist.

**Symptom:** `AttributeError: 'TokenizerWrapper' has no attribute 'convert_ids_to_tokens'`

**Fix:** Den inneren HF-Tokenizer extrahieren:
```python
# In json_logits_processor.py, _get_or_create_outlines_processor():
# Falls tokenizer ein Wrapper ist:
inner_tokenizer = getattr(tokenizer, '_tokenizer', tokenizer)
processor = JSONLogitsProcessor(schema_str, inner_tokenizer)
```

---

## Datenfluss-Diagramm

```
Client Request (response_format.type = "json_schema")
    │
    ▼
server.py ─── extract_json_schema() ──► json_schema dict
    │
    ▼ chat_kwargs["json_schema"] = schema
engine/batched.py (oder vlm.py)
    │
    ▼ SamplingParams(json_schema=schema)
scheduler.py :: _build_sampler_and_processors()
    │
    ▼ OutlinesJSONLogitsProcessor(schema, tokenizer)
    │
    ▼ logits_processors.append(json_processor)
    │
    ▼ BatchGenerator receives logits_processors
    │
    ▼ mlx-lm generate_step loop:
    │   for processor in logits_processors:
    │       logits = processor(tokens, logits)  ◄── Outlines masks here
    │
    ▼ Only schema-valid tokens survive sampling
    │
    ▼ Guaranteed valid JSON output
```

---

## Test-Matrix für Codex

| # | Test | Erwartetes Ergebnis | Priorität |
|---|------|---------------------|-----------|
| 1 | Request mit `json_schema` + Outlines installiert | Valides JSON, Schema-konform | P0 |
| 2 | Request mit `json_schema` + Outlines NICHT installiert | Fallback auf Prompt-Injection, Warning im Log | P0 |
| 3 | Request mit `json_object` (ohne Schema) | Bestehendes Verhalten, kein Outlines | P1 |
| 4 | Request ohne `response_format` | Bestehendes Verhalten, unverändert | P1 |
| 5 | Streaming + `json_schema` | Tokens streamen, finales JSON valide | P1 |
| 6 | Zwei parallele Requests: einer mit, einer ohne Schema | Beide korrekt, keine Cross-Contamination | P2 |
| 7 | Zweiter Request mit gleichem Schema | Instant (Cache-Hit), kein Re-Compile | P2 |
| 8 | VLM-Request mit Bild + `json_schema` | Vision + Structured Output zusammen | P3 |
| 9 | Anthropic API Path mit `text.format.json_schema` | Gleich wie OpenAI Path | P2 |
| 10 | Ungültiges Schema (kein valides JSON Schema) | Sauberer Error, kein Crash | P2 |

---

## Referenz-Links

- **oMLX Upstream:** https://github.com/jundot/omlx
- **Unser Fork:** https://github.com/inventstudio/omlx
- **Feature Branch:** https://github.com/inventstudio/omlx/tree/feature/outlines-structured-output
- **Outlines MLX Docs:** https://dottxt-ai.github.io/outlines/latest/features/models/mlxlm/
- **LM Studio's Outlines-Ansatz (Referenz):** https://lmstudio.ai/blog/lmstudio-v0.3.4
- **ThinkingBudgetProcessor (Pattern-Vorlage):** `omlx/api/thinking.py` Zeile 201ff

---

## Zusammenfassung für Codex

**Dein Job:** Klone den Fork, installier mit `[structured]`, teste die 10 Cases oben. Höchste Priorität hat Risiko 1 (mx.array vs torch.Tensor Kompatibilität). Wenn das klappt, klappt alles. Wenn nicht, bau den numpy-Konvertierungs-Wrapper ein.

**Nicht anfassen:** Die bestehende `_inject_json_instruction()` Logik in server.py. Die bleibt aktiv als zusätzliche Sicherheitsschicht neben dem Token-Enforcement.

**Ziel:** `response_format: json_schema` soll in oMLX genauso zuverlässig funktionieren wie in LM Studio — aber mit Continuous Batching, SSD-Cache und Multi-Model-Support.
