from typing import List, Dict, Any
import pandas as pd


def predict_texts(
    clf,
    texts: List[str],
    multi_label: bool,
    threshold: float,
    top_k: int,
    max_length: int,
    truncation: bool,
    pipeline_batch_size: int = 32,
) -> List[Dict[str, Any]]:
    if not texts:
        return []

    threshold = max(0.0, min(1.0, float(threshold)))
    top_k = max(1, int(top_k))
    max_length = max(8, int(max_length))
    pipeline_batch_size = max(1, int(pipeline_batch_size))

    tokenizer_kwargs = {"truncation": truncation, "max_length": max_length}
    apply_fn = "sigmoid" if multi_label else "softmax"

    if multi_label:
        raw = clf(
            texts,
            return_all_scores=True,
            batch_size=pipeline_batch_size,
            function_to_apply=apply_fn,
            **tokenizer_kwargs,
        )
    else:
        raw = clf(
            texts,
            top_k=top_k,
            batch_size=pipeline_batch_size,
            function_to_apply=apply_fn,
            **tokenizer_kwargs,
        )

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(texts):
        item = raw[i] if isinstance(raw, list) and i < len(raw) else []
        if multi_label:
            scores = item if isinstance(item, list) else []
            filtered = [s for s in scores if float(s.get("score", 0.0)) >= threshold]
            if not filtered and scores:
                best = max(scores, key=lambda s: float(s.get("score", 0.0)))
                filtered = [best]
            if not filtered:
                filtered = [{"label": "UNKNOWN", "score": 0.0}]
            filtered.sort(key=lambda s: float(s["score"]), reverse=True)
            preds = [
                {"label": s["label"], "score": round(float(s["score"]), 6)}
                for s in filtered
            ]
        else:
            preds_raw = (
                item
                if isinstance(item, list)
                else ([item] if isinstance(item, dict) else [])
            )
            if not preds_raw:
                preds_raw = [{"label": "UNKNOWN", "score": 0.0}]
            preds_raw.sort(key=lambda s: float(s["score"]), reverse=True)
            preds = [
                {"label": p["label"], "score": round(float(p["score"]), 6)}
                for p in preds_raw[:top_k]
            ]
        results.append({"text": text, "predictions": preds})

    return results


def format_results_wide(results: List[Dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    all_labels = sorted({p["label"] for r in results for p in r["predictions"]})
    rows = []
    for r in results:
        row = {"text": r["text"]}
        for lbl in all_labels:
            row[f"score__{lbl}"] = 0.0
        for p in r["predictions"]:
            row[f"score__{p['label']}"] = float(p["score"])
        rows.append(row)
    return pd.DataFrame(rows)
