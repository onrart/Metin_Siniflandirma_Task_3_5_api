import os
import json
from typing import Optional, Tuple, List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    pipeline,
)
from huggingface_hub import snapshot_download

# =========================
# models.json okuma & cache
# =========================
CONFIG_CACHE: Dict[str, dict] = {}
CONFIG_MTIME: Dict[str, float] = {}


def load_model_config(path: str) -> dict:
    """MODELS JSON'u diskten okur ve mtime'a göre basit cache uygular."""
    mtime = os.path.getmtime(path)
    if path in CONFIG_CACHE and CONFIG_MTIME.get(path) == mtime:
        return CONFIG_CACHE[path]
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    CONFIG_CACHE[path] = cfg
    CONFIG_MTIME[path] = mtime
    return cfg


def auto_device_index(device_index: Optional[int]) -> int:
    """
    device_index None (veya JSON'da yok) ise:
      - GPU varsa: 0 (ilk GPU)
      - değilse:  -1 (CPU)
    Aksi halde verilen değeri döndürür.
    """
    if device_index is None:
        return 0 if torch.cuda.is_available() else -1
    return device_index


def get_model_entry_by_key(key: str) -> tuple[str, Optional[str], int]:
    """
    models.json içinden key'e göre (repo_id, revision, device_index) döndürür.
    device_index yoksa otomatik GPU/CPU seçimi yapılır.
    """
    cfg_path = os.getenv("MODEL_CONFIG_PATH", "./models.json")
    cfg = load_model_config(cfg_path)
    models = cfg.get("models", {})
    if key not in models:
        raise ValueError(f"Model key not found: {key}")
    entry = models[key]
    repo_id: str = entry["repo_id"]
    revision: Optional[str] = entry.get("revision")
    device_index: int = auto_device_index(entry.get("device_index"))
    return repo_id, revision, device_index


def get_default_model_from_config() -> tuple[str, Optional[str], int]:
    """models.json'daki default_key'e göre varsayılan modeli döndürür."""
    cfg_path = os.getenv("MODEL_CONFIG_PATH", "./models.json")
    cfg = load_model_config(cfg_path)
    default_key = cfg.get("default_key")
    if not default_key:
        raise ValueError("default_key missing in models.json")
    return get_model_entry_by_key(default_key)


# =========================
# Cihaz / Yol yardımcıları
# =========================


def resolve_device(device_index: int) -> int:
    """
    Geçerli bir pipeline cihaz indexi döndürür:
    - CUDA varsa ve device_index 0..cuda_count-1 ise -> kendisi
    - device_index -1 ise -> CPU
    - aksi halde -> CPU
    """
    if device_index == -1:
        return -1
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if 0 <= device_index < cuda_count:
            return device_index
    return -1  # CPU fallback


def _is_local_path(path: str) -> bool:
    return os.path.isdir(path)


def _resolve_model_source(model_path: str, revision: Optional[str]) -> str:
    """
    HF repo id veya yerel klasör olabilir.
      - Yerel klasörse doğrudan onu döndürür.
      - Repo id ise:
          - revision varsa snapshot indir ve yerel yolu döndür (sabit sürüm).
          - yoksa repo id'yi döndür (transformers otomatik indir/cache'ler).
    """
    if _is_local_path(model_path):
        return model_path
    if revision:
        local_path = snapshot_download(repo_id=model_path, revision=revision)
        return local_path
    return model_path


# =========================
# Sınıflandırıcı holder (cache)
# =========================
class ClassifierHolder:
    """
    Tek-aktif model yaklaşımı:
      - clf/tokenizer cache
      - etiket listesi
      - device bilgisi
      - signature: (path, device_idx, revision)
    """

    def __init__(self) -> None:
        self.clf = None
        self.labels: List[str] = []
        self.device_str: Optional[str] = None
        self.signature: Optional[tuple] = None  # (path, device_idx, revision)

    def unload(self):
        """Cache ve GPU RAM boşalt."""
        self.clf = None
        self.labels = []
        self.device_str = None
        self.signature = None
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def load(self, model_path: str, device_index: int, revision: Optional[str]):
        """
        Modeli yükler; signature aynıysa tekrar yüklemez.
        dtype='auto' kullanır (GPU'da fp16/bf16, CPU'da fp32).
        """
        dev_idx = resolve_device(device_index)
        sig = (model_path, dev_idx, revision)
        if self.signature == sig and self.clf is not None:
            return self.clf, self.labels, self.device_str

        self.unload()

        model_to_load = _resolve_model_source(model_path, revision)

        # Tokenizer & Model
        tok = AutoTokenizer.from_pretrained(model_to_load)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_to_load,
            dtype="auto",  # torch_dtype deprecated → dtype
        )
        mdl.eval()

        # Pipeline
        clf = pipeline(
            task="text-classification",
            model=mdl,
            tokenizer=tok,
            device=dev_idx,
        )

        # Etiketler
        labels: List[str] = []
        try:
            cfg = AutoConfig.from_pretrained(model_to_load)
            id2label = getattr(cfg, "id2label", None)
            if isinstance(id2label, dict) and len(id2label) > 0:
                labels = [id2label[i] for i in sorted(map(int, id2label.keys()))]
        except Exception:
            pass

        # Warmup (yalnızca GPU)
        try:
            if dev_idx != -1:
                _ = clf(["warmup"], top_k=1, truncation=True, max_length=32)
        except Exception:
            pass

        self.clf = clf
        self.labels = labels
        self.device_str = "cuda" if dev_idx != -1 else "cpu"
        self.signature = sig
        return self.clf, self.labels, self.device_str


# Singleton
HOLDER = ClassifierHolder()
