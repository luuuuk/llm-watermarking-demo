import hashlib
import math
import threading
import uuid

import torch
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_K = 8
MAX_NEW_TOKENS = 200

TOURNAMENT_DEPTH = 3  # 2**TOURNAMENT_DEPTH candidates compete
MAX_CONTEXT = 16  # context tokens stored per generated token for re-detection
MAX_SESSIONS = 64
MODES = ("none", "hard", "soft", "tournament")

app = Flask(__name__, static_folder="static", static_url_path="")

_tokenizer = None
_model = None
_sessions = {}
_sessions_lock = threading.Lock()


def _load_model():
    """Lazily load the model/tokenizer once and reuse across requests."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _key_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _green_mask(key, ctx, vocab_size, gamma):
    """Kirchenbauer-style green list: a context-seeded permutation of the vocabulary."""
    h = hashlib.sha256(key.encode())
    for cid in ctx:
        h.update(int(cid).to_bytes(4, "big"))
    gen = torch.Generator().manual_seed(int.from_bytes(h.digest()[:7], "big"))
    perm = torch.randperm(vocab_size, generator=gen)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[perm[: int(gamma * vocab_size)]] = True
    return mask


def _g_value(seed, layer, context_ids, token_id):
    """Deterministic pseudorandom float in [0, 1) keyed by watermark seed + context + candidate."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(8, "big"))
    h.update(layer.to_bytes(4, "big"))
    for cid in context_ids:
        h.update(int(cid).to_bytes(4, "big"))
    h.update(int(token_id).to_bytes(4, "big"))
    return int.from_bytes(h.digest()[:8], "big") / 2**64


def _stats(greens, n, gamma):
    if n == 0:
        return {"n": 0, "greens": 0, "gamma": gamma, "z": 0.0, "p": 1.0}
    var = n * gamma * (1 - gamma)
    z = (greens - gamma * n) / math.sqrt(var) if var > 0 else 0.0
    return {
        "n": n,
        "greens": greens,
        "gamma": gamma,
        "z": z,
        "p": 0.5 * math.erfc(z / math.sqrt(2)),
    }


def _score(session, results, h, gamma):
    """Z-test over unique (context, token) pairs; repeats are not independent trials. EOS is skipped.

    Each result is (green, hits, trials): one Bernoulli trial per list check, or one per tournament round.
    """
    eos = _load_model()[0].eos_token_id
    seen = set()
    hits = trials = 0
    for tok, ctx, res in zip(session["generated_tokens"], session["token_ctx"], results):
        pair = (tuple(ctx[-h:]), tok["id"])
        if tok["id"] == eos or pair in seen:
            continue
        seen.add(pair)
        hits += res[1]
        trials += res[2]
    return _stats(hits, trials, gamma)


def _session_stats(session):
    gamma = 0.5 if session["mode"] == "tournament" else session["gamma"]
    results = [(t["green"], t["hits"], t["trials"]) for t in session["generated_tokens"]]
    return _score(session, results, session["h"], gamma)


def _candidate_color(session, mask, token_id):
    if session["mode"] == "tournament":
        green, _, _, gs = _classify("tournament", session["key"], 0.5, session["last_ctx"], token_id, 0)
        return {"green": green, "g": gs}
    return {"green": bool(mask[token_id])}


def _prepare_next(session, probs):
    """Compute the green list and the watermark-adjusted distribution for the next token."""
    tokenizer, _ = _load_model()
    ctx = session["generated_ids"][0, -session["h"] :].tolist()
    mask = _green_mask(session["key"], ctx, probs.shape[-1], session["gamma"])
    gen_mask = mask & ~session["forced_mask"]
    gen_mask[tokenizer.eos_token_id] = True  # never ban the stop token

    adj = probs
    if session["mode"] == "hard":
        masked = probs * gen_mask
        if masked.sum() > 0:
            adj = masked / masked.sum()
    elif session["mode"] == "soft":
        adj = probs * torch.exp(session["delta"] * gen_mask.float())
        adj = adj / adj.sum()

    session["last_probs"] = probs
    session["adj_probs"] = adj
    session["last_mask"] = mask
    session["last_ctx"] = ctx

    top_probs, top_ids = torch.topk(adj, k=TOP_K)
    result = []
    for tid, p in zip(top_ids, top_probs):
        item = {
            "id": int(tid),
            "token": tokenizer.decode([int(tid)]),
            "prob": float(p),
            "base_prob": float(probs[tid]),
        }
        if session["mode"] != "none":
            item.update(_candidate_color(session, mask, int(tid)))
        result.append(item)
    return result


def _forward(session, input_ids):
    _, model = _load_model()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=session["past_key_values"],
            use_cache=True,
        )
    session["past_key_values"] = outputs.past_key_values
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1).squeeze(0)
    return _prepare_next(session, probs)


def _run_tournament(session, tokenizer):
    """Sample 2**depth candidates and knock them out round by round using g-values."""
    probs = session["last_probs"]
    seed = _key_seed(session["key"])
    n_candidates = 2**TOURNAMENT_DEPTH
    candidate_ids = torch.multinomial(probs, n_candidates, replacement=True).tolist()
    context_ids = session["last_ctx"]

    current = [
        {"id": tid, "token": tokenizer.decode([tid]), "prob": float(probs[tid])}
        for tid in candidate_ids
    ]
    rounds = []
    layer = 0
    while len(current) > 1:
        matches = []
        next_round = []
        for a, b in zip(current[0::2], current[1::2]):
            ga = _g_value(seed, layer, context_ids, a["id"])
            gb = _g_value(seed, layer, context_ids, b["id"])
            winner_key = "a" if ga >= gb else "b"
            matches.append(
                {"a": {**a, "g": ga}, "b": {**b, "g": gb}, "winner": winner_key}
            )
            next_round.append(a if winner_key == "a" else b)
        rounds.append(matches)
        current = next_round
        layer += 1

    return rounds, current[0]


def _classify(mode, key, gamma, ctx, token_id, vocab_size, mask=None):
    """Returns (green, hits, trials, per-round g-values or None)."""
    if mode == "tournament":
        seed = _key_seed(key)
        gs = [_g_value(seed, layer, ctx, token_id) for layer in range(TOURNAMENT_DEPTH)]
        hits = sum(g > 0.5 for g in gs)
        return hits * 2 > TOURNAMENT_DEPTH, hits, TOURNAMENT_DEPTH, gs
    if mask is None:
        mask = _green_mask(key, ctx, vocab_size, gamma)
    green = bool(mask[token_id])
    return green, int(green), 1, None


def _commit_token(session, tokenizer, token_id):
    """Append the chosen token to the session and compute the following distribution."""
    token_id = int(token_id)
    green, hits, trials, gs = _classify(
        session["mode"],
        session["key"],
        session["gamma"],
        session["last_ctx"],
        token_id,
        session["last_probs"].shape[-1],
        session["last_mask"],
    )
    session["generated_tokens"].append(
        {
            "id": token_id,
            "token": tokenizer.decode([token_id]),
            "prob": float(session["last_probs"][token_id]),
            "adj_prob": float(session["adj_probs"][token_id]),
            "green": green,
            "hits": hits,
            "trials": trials,
            "g": gs,
        }
    )
    session["token_ctx"].append(session["generated_ids"][0, -MAX_CONTEXT:].tolist())

    next_token = torch.tensor([[token_id]])
    session["generated_ids"] = torch.cat([session["generated_ids"], next_token], dim=-1)
    session["steps"] += 1

    done = (
        token_id == tokenizer.eos_token_id or session["steps"] >= session["max_new"]
    )
    session["done"] = done
    distribution = [] if done else _forward(session, next_token)
    return _snapshot(session, tokenizer, distribution)


def _snapshot(session, tokenizer, distribution):
    return {
        "distribution": distribution,
        "generated_text": tokenizer.decode(
            session["generated_ids"][0, session["prompt_len"] :],
            skip_special_tokens=True,
        ),
        "generated_tokens": list(session["generated_tokens"]),
        "stats": _session_stats(session),
        "done": session["done"],
    }


def _sample_step(session, tokenizer):
    """Advance one token using the session's watermark mode; returns (result, tournament)."""
    if session["mode"] == "tournament":
        rounds, final = _run_tournament(session, tokenizer)
        return _commit_token(session, tokenizer, final["id"]), rounds, final
    token_id = int(torch.multinomial(session["adj_probs"], 1))
    return _commit_token(session, tokenizer, token_id), None, None


def _forced_mask(tokenizer, words, vocab_size):
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for word in words:
        variants = {word, " " + word, word.capitalize(), " " + word.capitalize()}
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if ids:
                mask[ids[0]] = True  # first sub-token stands in for the whole word
    return mask


def _clamp(value, lo, hi, default):
    try:
        return min(max(float(value), lo), hi)
    except (TypeError, ValueError):
        return default


def _get_session(data):
    with _sessions_lock:
        return _sessions.get(data.get("session_id"))


@app.after_request
def _allow_cors(response):
    # Lets a Svelte app fetch the API directly, not just via same-origin iframe.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/start", methods=["POST", "OPTIONS"])
def start():
    if request.method == "OPTIONS":
        return "", 204

    tokenizer, _ = _load_model()
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "Why is the sky blue?")
    mode = data.get("mode", "none")
    if mode not in MODES:
        return jsonify(error="invalid mode"), 400
    words = [w.strip() for w in str(data.get("red_words", "")).split(",") if w.strip()]

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    session = {
        "mode": mode,
        "gamma": _clamp(data.get("gamma"), 0.05, 0.95, 0.5),
        "delta": _clamp(data.get("delta"), 0.0, 20.0, 2.0),
        "h": int(_clamp(data.get("h"), 1, 8, 4)),
        "key": str(data.get("key") or "your_token"),
        "max_new": int(_clamp(data.get("max_new"), 1, 400, MAX_NEW_TOKENS)),
        "generated_ids": input_ids,
        "prompt_len": input_ids.shape[-1],
        "past_key_values": None,
        "generated_tokens": [],
        "token_ctx": [],
        "steps": 0,
        "done": False,
    }
    _, model = _load_model()
    session["forced_mask"] = _forced_mask(
        tokenizer, words, model.get_output_embeddings().weight.shape[0]
    )
    distribution = _forward(session, input_ids)

    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = session
        while len(_sessions) > MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))

    return jsonify(session_id=session_id, **_snapshot(session, tokenizer, distribution))


@app.route("/api/step", methods=["POST", "OPTIONS"])
def step():
    if request.method == "OPTIONS":
        return "", 204

    tokenizer, _ = _load_model()
    data = request.get_json(silent=True) or {}
    session = _get_session(data)
    token_id = data.get("token_id")
    if session is None:
        return jsonify(error="unknown session_id"), 404
    if session["done"]:
        return jsonify(error="session finished"), 400
    if token_id is None:
        return jsonify(error="token_id is required"), 400

    result = _commit_token(session, tokenizer, token_id)
    return jsonify(session_id=data["session_id"], **result)


@app.route("/api/sample", methods=["POST", "OPTIONS"])
def sample():
    if request.method == "OPTIONS":
        return "", 204

    tokenizer, _ = _load_model()
    data = request.get_json(silent=True) or {}
    session = _get_session(data)
    if session is None:
        return jsonify(error="unknown session_id"), 404
    if session["done"]:
        return jsonify(error="session finished"), 400

    result, rounds, final = _sample_step(session, tokenizer)
    return jsonify(
        session_id=data["session_id"],
        tournament=rounds,
        selected_token=final,
        **result,
    )


@app.route("/api/auto", methods=["POST", "OPTIONS"])
def auto():
    if request.method == "OPTIONS":
        return "", 204

    tokenizer, _ = _load_model()
    data = request.get_json(silent=True) or {}
    session = _get_session(data)
    if session is None:
        return jsonify(error="unknown session_id"), 404

    result = _snapshot(session, tokenizer, [])
    while not session["done"]:
        result, _, _ = _sample_step(session, tokenizer)
    return jsonify(session_id=data["session_id"], **result)


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    session = _get_session(data)
    if session is None:
        return jsonify(error="unknown session_id"), 404

    key = str(data.get("key") or session["key"])
    mode = data.get("mode", session["mode"])
    gamma = _clamp(data.get("gamma"), 0.05, 0.95, session["gamma"])
    h = int(_clamp(data.get("h"), 1, 8, session["h"]))
    vocab = session["last_probs"].shape[-1]

    flags = []
    results = []
    for tok, ctx in zip(session["generated_tokens"], session["token_ctx"]):
        green, hits, trials, _ = _classify(mode, key, gamma, ctx[-h:], tok["id"], vocab)
        flags.append(green)
        results.append((green, hits, trials))

    det_gamma = 0.5 if mode == "tournament" else gamma
    return jsonify(green_flags=flags, stats=_score(session, results, h, det_gamma))


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
