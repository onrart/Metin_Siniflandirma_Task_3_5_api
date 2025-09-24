import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import HOLDER, get_default_model_from_config, get_model_entry_by_key
from .inference import predict_texts
from .schemas import (
    SinglePredictRequest,
    BatchPredictRequest,
    SinglePredictResponse,
    BatchPredictResponse,
    PredictionItem,
)

API_KEY = os.getenv("API_KEY")

def require_key(authorization: str | None):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="Text Classification API", version="1.0")

# ---- CORS ----
origins_raw = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_load_model():
    repo_id, revision, device_idx = get_default_model_from_config()
    HOLDER.load(repo_id, device_idx, revision)



@app.get("/")
async def root():
    return {"ok": True, "see": "/docs"}


@app.get("/health")
async def health():
    ok = HOLDER.clf is not None
    return {"status": "ok" if ok else "cold", "device": HOLDER.device_str}


@app.get("/labels")
async def labels():
    return {"labels": HOLDER.labels}


@app.get("/current-model")
async def current_model():
    return {
        "signature": getattr(HOLDER, "signature", None),
        "device": HOLDER.device_str,
        "labels": HOLDER.labels,
    }


@app.post("/load-model-by-key")
async def load_model_by_key(req: dict, authorization: str | None = Header(None)):
    require_key(authorization)
    key = req.get("key")
    if not key:
        raise HTTPException(status_code=400, detail="Missing 'key'")
    repo_id, revision, device_idx = get_model_entry_by_key(key)
    clf, labels, device = HOLDER.load(repo_id, device_idx, revision)
    return {
        "status": "loaded",
        "key": key,
        "repo_id": repo_id,
        "revision": revision,
        "device": device,
        "labels": labels,
    }


@app.post("/predict", response_model=SinglePredictResponse)
async def predict(req: SinglePredictRequest, authorization: str | None = Header(None)):
    # İstersen auth buraya da ekleyebilirsin: require_key(authorization)
    if HOLDER.clf is None:
        raise HTTPException(status_code=503, detail="Model hazır değil")
    res = predict_texts(
        HOLDER.clf,
        [req.text],
        multi_label=req.params.multi_label,
        threshold=req.params.threshold,
        top_k=req.params.top_k,
        max_length=req.params.max_length,
        truncation=req.params.truncation,
        pipeline_batch_size=req.params.pipeline_batch_size,
    )
    r0 = res[0]
    return SinglePredictResponse(
        text=r0["text"],
        predictions=[PredictionItem(**p) for p in r0["predictions"]],
    )


@app.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(
    req: BatchPredictRequest, authorization: str | None = Header(None)
):
    # İstersen auth buraya da ekleyebilirsin: require_key(authorization)
    if HOLDER.clf is None:
        raise HTTPException(status_code=503, detail="Model hazır değil")
    res = predict_texts(
        HOLDER.clf,
        req.texts,
        multi_label=req.params.multi_label,
        threshold=req.params.threshold,
        top_k=req.params.top_k,
        max_length=req.params.max_length,
        truncation=req.params.truncation,
        pipeline_batch_size=req.params.pipeline_batch_size,
    )
    return BatchPredictResponse(
        results=[
            SinglePredictResponse(
                text=r["text"],
                predictions=[PredictionItem(**p) for p in r["predictions"]],
            )
            for r in res
        ]
    )


