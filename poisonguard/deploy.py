"""poisonguard deploy：把训练产物打包成单文件发布物，离线评分不入域。

背景
----
在线(service)与批扫(scan)都需要"装配"：
    build_scorer(cfg) -> enc + knn + ngram + cal + student
而装配依赖 train/dev 原文本与编码器 —— 部署侧往往没有这些，也不应该把
用户原始语料发给下游。于是"发布物"应当包含的是**拟合态**而不是数据：
    - kNN 近邻索引(参考句向量，非原文)
    - 字符 n-gram 分类器 + 向量化器
    - 校准分布(参考分数)
    - 蒸馏学生模型
    - 该域运营阈值(学生阈值 + 判官 logprob 阈值)
打包成**单个 pickle(.pkg)**。运行时只需公开的编码器权重(公共模型)，
不需要原始文本、不需要 dev/test、不需要判官大模型 —— "数据不出域"成立。

两个命令(挂进 cli)：
    poisonguard deploy --domain afqmc            # 作者侧：打 .pkg
    poisonguard run --bundle x.pkg --input f.txt # 部署侧：离线批扫(复用 scan 报告)

一致性保证：
    - .pkg 记录 encoder.name 与 max_length，运行时必须用同一编码器重编码
      (换编码器 = 换特征空间，kNN / 学生就失效，故意在加载时校验并报错)。
    - JudgeStudent / CharNgramDetector / Calibration 均为纯 sklearn/标准库对象，
      可 pickle；`score()` 语义与 service.PoisonScorer.score 完全一致。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import os
import json
import pickle
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .detectors import Calibration, CharNgramDetector, KNNDetector
from .service import build_scorer

PKG_VERSION = 1


# ----------------------------------------------------------------------
@dataclass
class DeployState:
    """在 .pkg 里实际落盘的内容(全部为拟合态, 不含原始文本)。"""

    version: int = PKG_VERSION
    domain: str = ""
    encoder: dict[str, Any] = field(default_factory=dict)   # {name, max_length, ...}
    detector: dict[str, Any] = field(default_factory=dict)  # {target_fpr, knn_k}
    threshold: float = 0.5
    judge_threshold: float = 0.5
    knn: KNNDetector | None = None
    ngram: CharNgramDetector | None = None
    cal: Calibration | None = None
    student: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployedScorer:
    """轻量运行时评分器：与 service.PoisonScorer.score 同输出形状。"""

    state: DeployState
    enc: Any

    # 与 PoisonScorer 对齐：学生来自包内(state.student)，没有外部 pkl 路径
    student_path = "bundled"

    @property
    def version(self) -> str:
        return f"pkg-v{PKG_VERSION}/{self.state.domain}"

    @property
    def threshold(self) -> float:
        return self.state.threshold

    @property
    def has_student(self) -> bool:
        return self.state.student is not None

    @property
    def judge_threshold(self) -> float:
        return self.state.judge_threshold

    @property
    def domain(self) -> str:
        """包内领域名（报告口径用；部署机没有 processed_<域> 目录）。"""
        return self.state.domain or "bundled"

    @property
    def encoder_name(self) -> str:
        return (self.state.encoder or {}).get("name", "?")

    def score(self, texts: list[str], explain: bool = False) -> dict:
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return {"results": [], "n": 0, "latency_ms": 0.0,
                    "threshold": self.threshold, "model": self.version}
        t0 = time.perf_counter()
        X = self.enc.encode(texts)
        s = self.state
        knn = s.knn.score(X, texts) if s.knn else np.zeros(len(texts))
        fpr = s.cal.to_fpr(knn) if s.cal else np.full(len(texts), np.nan)
        ngram = s.ngram.score(X, texts) if s.ngram else np.full(len(texts), np.nan)
        stu = s.student.score(texts, X) if s.student else np.full(len(texts), np.nan)

        out = []
        for i, t in enumerate(texts):
            if np.isfinite(stu[i]):
                v = float(stu[i])
            elif np.isfinite(fpr[i]):
                v = float(1.0 - fpr[i])
            else:
                v = float(knn[i])
            rec = {"score": round(v, 6), "flag": bool(v > self.threshold)}
            if explain:
                rec["views"] = {
                    "knn": round(float(knn[i]), 6),
                    "fpr_est": None if not np.isfinite(fpr[i]) else round(float(fpr[i]), 6),
                    "char_ngram": None if not np.isfinite(ngram[i]) else round(float(ngram[i]), 6),
                    "student": None if not np.isfinite(stu[i]) else round(float(stu[i]), 6),
                    "chars": len(t),
                }
            out.append(rec)
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "results": out, "n": len(out),
            "latency_ms": round(dt, 3),
            "ms_per_item": round(dt / max(len(out), 1), 4),
            "threshold": self.threshold,
            "model": self.version,
        }


# ----------------------------------------------------------------------
def _domain_tag(cfg: Config) -> str:
    """processed 目录后缀中的领域名：processed_afqmc -> afqmc / processed_real_v3 -> real_v3。

    注意与 student 文件名区分：学生模型是 `student_judge{tag}.pkl`（tag 含前导下划线），
    而 judge 指标与 .pkg 文件名用**去掉前导下划线**的领域名（judge_afqmc.json）。
    """
    tag = Path(cfg.data.processed_dir).name[len("processed"):]
    return tag.lstrip("_")


def _student_path_for(cfg: Config, fixed: str | Path | None = None) -> Path | None:
    mdir = Path(cfg.metrics_dir)
    tag = Path(cfg.data.processed_dir).name[len("processed"):]
    cands = []
    if fixed is not None:
        cands.append(Path(fixed))
    cands += [mdir / f"student_judge{tag}.pkl",
              *sorted(mdir.glob("student_judge*.pkl"))]
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return None


def _load_judge_threshold(cfg: Config) -> float:
    mdir = Path(cfg.metrics_dir)
    domain = _domain_tag(cfg)
    for cand in (mdir / f"judge_{domain}.json", *sorted(mdir.glob("judge_*.json"))):
        if not cand.exists():
            continue
        try:
            j = json.loads(cand.read_text(encoding="utf-8"))
            for _name, det in (j.get("detectors") or {}).items():
                thr = det.get("threshold")
                if thr is not None:
                    return float(thr)
        except Exception:
            continue
    return 0.5


class PackedKNN:
    """量化后的 kNN 参考集：只存码字+缩放，运行时分块重建做余弦近邻。

    全精度 float32 参考向量是 .pkg 体积的主要来源（13807×768×4B ≈ 42MB）。
    - ``f16``：按行 half 存储，体积约减半，余弦距离误差 ~1e-4；
    - ``i8`` ：按行（L2 归一化后）对称量化，体积约 1/4，误差 ~1e-3。
    语义与 ``detectors.KNNDetector.score`` 相同：到最近 k 个干净样本的平均余弦距离。
    """

    name = "knn"

    def __init__(self, codes: np.ndarray, scale: np.ndarray, k: int,
                 dtype: str = "f16", chunk: int = 2048):
        self.codes = codes
        self.scale = scale
        self.k = int(k)
        self.dtype = dtype
        self.chunk = int(chunk)

    @classmethod
    def from_knn(cls, knn: KNNDetector, mode: str = "i8") -> "PackedKNN":
        X = np.asarray(knn.nn._fit_X, dtype=np.float32)
        k = int(getattr(knn, "k_eff", knn.k))
        if mode == "f16":
            return cls(X.astype(np.float16), np.float32(1.0), k, dtype="f16")
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Xn = X / norm
        scale = (np.abs(Xn).max(axis=1, keepdims=True) / 127.0).astype(np.float32)
        scale[scale == 0] = np.float32(1e-8)
        codes = np.clip(np.rint(Xn / scale), -127, 127).astype(np.int8)
        return cls(codes, scale, k, dtype="i8")

    def _chunk(self, i: int) -> np.ndarray:
        blk = self.codes[i:i + self.chunk]
        if self.dtype == "f16":
            return np.asarray(blk, dtype=np.float32)
        sc = np.asarray(self.scale[i:i + self.chunk], dtype=np.float32)
        return np.asarray(blk, dtype=np.float32) * sc

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Xn = X / nrm
        n = Xn.shape[0]
        k = max(1, min(self.k, len(self.codes)))
        best = np.full((n, 0), -np.inf, dtype=np.float32)
        for j in range(0, len(self.codes), self.chunk):
            sim = Xn @ self._chunk(j).T              # 余弦相似度
            merged = np.concatenate([best, sim], axis=1)
            kk = min(k, merged.shape[1])
            best = np.partition(merged, merged.shape[1] - kk, axis=1)[:, -kk:]
        return (1.0 - best).mean(axis=1).astype(np.float32)


def _quantize_knn(knn: KNNDetector | None, mode: str) -> Any:
    """按 mode 把 kNN 参考集压成 f16/int8 码字（none 原样返回）。"""
    if knn is None or mode in ("", "none", None):
        return knn
    if mode not in ("f16", "i8"):
        raise ValueError(f"未知量化模式: {mode}（可选 none/f16/i8）")
    return PackedKNN.from_knn(knn, mode=mode)


ENV_SIGN_KEY = "POISONGUARD_PKG_KEY"


def _resolve_sign_key(key: str | None = None,
                      key_file: str | Path | None = None) -> str | None:
    """签名密钥优先级：显式 --sign-key > --sign-key-file > 环境变量。"""
    if key:
        return str(key)
    if key_file:
        return Path(key_file).read_text(encoding="utf-8").strip() or None
    env = os.environ.get(ENV_SIGN_KEY, "").strip()
    return env or None


def _container_core(container: dict) -> bytes:
    """被签名的规范化字节串（不含 sig 本身）：format|version|sha256|manifest。"""
    manifest = json.dumps(container.get("manifest") or {}, sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"))
    return (f"{container.get('format')}|{container.get('version')}|"
            f"{container.get('sha256')}|{manifest}").encode("utf-8")


def _sign_container(container: dict, key: str) -> str:
    return hmac.new(key.encode("utf-8"), _container_core(container),
                    hashlib.sha256).hexdigest()


# ---------------- ed25519（非对称签名：作者私钥签、部署方公钥验） ----------------
def _load_ed25519_private(path: str | Path):
    """读私钥：支持 PEM 与裸 32 字节 seed 的 hex/base64。"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ed25519 私钥不存在: {p}")
    raw = p.read_bytes()
    if b"-----BEGIN" in raw:
        key = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("私钥不是 ed25519")
        return key
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        seed = bytes.fromhex(text)
    except ValueError:
        import base64

        seed = base64.b64decode(text)
    if len(seed) != 32:
        raise ValueError(f"ed25519 私钥应是 32 字节（hex 64 字符），实际 {len(seed)} 字节")
    return Ed25519PrivateKey.from_private_bytes(seed)


def _load_ed25519_public(path: str | Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ed25519 公钥不存在: {p}")
    raw = p.read_bytes()
    if b"-----BEGIN" in raw:
        key = serialization.load_pem_public_key(raw)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("公钥不是 ed25519")
        return key
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        seed = bytes.fromhex(text)
    except ValueError:
        import base64

        seed = base64.b64decode(text)
    if len(seed) != 32:
        raise ValueError(f"ed25519 公钥应是 32 字节（hex 64 字符），实际 {len(seed)} 字节")
    return Ed25519PublicKey.from_public_bytes(seed)


def _sign_container_ed25519(container: dict, priv) -> str:
    return priv.sign(_container_core(container)).hex()


def _verify_container_ed25519(container: dict, sig_hex: str) -> bool:
    from cryptography.exceptions import InvalidSignature

    try:
        pub = container["_pub"]
        pub.verify(bytes.fromhex(sig_hex), _container_core(container))
        return True
    except (InvalidSignature, ValueError, KeyError):
        return False


def generate_ed25519_keypair(out_dir: str | Path, name: str = "author") -> tuple[Path, Path]:
    """生成作者密钥对：私钥 hex 落盘并尽量设为 0600，公钥可自由分发。"""
    from cryptography.hazmat.primitives import serialization

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    priv_path = out / f"{name}.key"
    pub_path = out / f"{name}.pub"
    priv_path.write_text(priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()).hex(), encoding="utf-8")
    pub_path.write_text(priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw).hex(), encoding="utf-8")
    try:
        os.chmod(priv_path, 0o600)
    except OSError:
        pass
    return priv_path, pub_path


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def encoder_fingerprint(cfg: Config) -> str:
    """编码器权重指纹：model_dir 下 *.safetensors/*.bin 的「路径+大小+sha256」聚合。

    用途是防「同名不同权重」——部署机装了同款模型的另一个版本时，
    kNN / 学生模型的特征空间就变了，分数不可比。16 位 hex（64 bit）
    足够日常运维区分，不是密码学身份（那属于签名该做的事）。
    """
    root = Path(getattr(cfg.encoder, "model_dir", "") or "")
    if not root.is_dir():
        return ""
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".safetensors", ".bin"))
    if not files:
        return ""
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}|{p.stat().st_size}|".encode("utf-8"))
        h.update(_hash_file(p).encode("ascii"))
    return h.hexdigest()[:16]


def bundle_scorer(cfg: Config, out_path: str | Path | None = None,
                  ed25519_key: str | Path | None = None,
                  student_path: str | Path | None = None,
                  threshold: float | None = None,
                  quantize: str = "none",
                  sign_key: str | None = None,
                  calibrate: bool = True) -> Path:
    """作者侧：装配该域打分器 → 提取拟合态 → 落单文件 .pkg。

    阈值口径（P0-1）：默认在 dev 干净样本上标定 1% 工作点，写进 manifest；
    `threshold` 显式给了就以它为准（人工覆盖）；`calibrate=False` 则沿用
    打分器自带阈值（与标定口径可能不一致，只建议 debug 用）。
    """
    from .threshold import clean_dev_texts, calibrate_threshold

    scorer = build_scorer(cfg, student_path=_student_path_for(cfg, student_path))
    threshold_source = "scorer_default"
    calib: dict | None = None
    if threshold is not None:
        scorer.threshold = float(threshold)
        threshold_source = "cli_override"
        print(f"[deploy] 阈值 {scorer.threshold:.4f}：命令行显式覆盖（非标定值）")
    elif calibrate:
        try:
            texts = clean_dev_texts(cfg)
        except Exception as exc:  # 没有 processed 目录时退回默认阈值
            print(f"[deploy] 警告: 取不到 dev 干净样本({exc})，跳过阈值标定")
            texts = []
        if texts:
            calib = calibrate_threshold(scorer, texts, cfg.detector.target_fpr)
            scorer.threshold = calib["threshold"]
            threshold_source = "calibrated"
            print(f"[deploy] 阈值 {scorer.threshold:.4f}：在 {calib['n_clean']} 条 dev "
                  f"干净样本上标定（目标 FPR {cfg.detector.target_fpr:.2%}，"
                  f"实测 {calib['achieved_fpr']:.2%}）")
    else:
        print(f"[deploy] 阈值 {scorer.threshold:.4f}：打分器默认值（未标定）")

    domain = _domain_tag(cfg)
    state = DeployState(
        domain=domain,
        encoder={
            "name": cfg.encoder.name,
            "max_length": cfg.encoder.max_length,
            "pooling": cfg.encoder.pooling if hasattr(cfg.encoder, "pooling") else "mean",
        },
        detector={"target_fpr": cfg.detector.target_fpr, "knn_k": cfg.detector.knn_k},
        threshold=float(scorer.threshold),
        judge_threshold=_load_judge_threshold(cfg),
        knn=_quantize_knn(scorer.knn, quantize),
        ngram=scorer.ngram,
        cal=scorer.cal,
        student=scorer.student,
        provenance={
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "encoder": cfg.encoder.name,
            "has_student": scorer.student is not None,
            "has_ngram": scorer.ngram is not None,
            "has_knn": scorer.knn is not None,
            "quantize": quantize if quantize not in ("", None) else "none",
            "knn_refs": int(len(getattr(scorer.knn, "nn", None)._fit_X))
            if getattr(scorer.knn, "nn", None) is not None else 0,
        },
    )
    fp = ""
    try:
        fp = encoder_fingerprint(cfg)
    except OSError as exc:  # 权重文件被占用/不可读时不阻塞打包
        print(f"[deploy] 警告: 编码器指纹计算失败({exc})，包内不带指纹")

    if out_path is None:
        out_path = Path("outputs") / "deploy" / f"poisonguard-{domain or 'default'}.pkg"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 容器格式：payload(拟合态 pickle) + sha256 + manifest → 部署侧可校验完整性
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    container = {
        "format": "poisonguard.pkg",
        "version": PKG_VERSION,
        "sha256": digest,
        "bytes": len(payload),
        "manifest": {
            "domain": domain,
            "encoder": cfg.encoder.name,
            "max_length": cfg.encoder.max_length,
            "quantize": state.provenance.get("quantize", "none"),
            "knn_refs": state.provenance.get("knn_refs", 0),
            "threshold": state.threshold,
            "threshold_source": threshold_source,
            "threshold_calibrated": calib,
            "judge_threshold": state.judge_threshold,
            "created": state.provenance.get("created"),
            "encoder_fingerprint": fp,
            "signed": ("ed25519" if ed25519_key else ("hmac" if sign_key else None)),
        },
        "payload": payload,
    }
    if ed25519_key:
        container["sig_ed25519"] = _sign_container_ed25519(
            container, _load_ed25519_private(ed25519_key))
    elif sign_key:
        container["sig"] = _sign_container(container, sign_key)
    with open(out_path, "wb") as f:
        pickle.dump(container, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[deploy] {domain} 拟合态已打包 -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB, 含原始文本: no, "
          f"量化: {container['manifest']['quantize']})")
    print(f"[deploy] sha256={digest[:16]}…  payload={len(payload) / 1024:.0f} KB  "
          f"knn_refs={container['manifest']['knn_refs']}")
    scheme = "ed25519 已签(作者私钥)" if ed25519_key else (
        "HMAC-SHA256 已签(共享密钥)" if sign_key else "未签名(仅完整性校验)")
    print(f"[deploy] 签名: {scheme}  "
          f"编码器指纹: {fp or '(未计算)'}")
    return out_path


def load_bundle(path: str | Path, cfg: Config | None = None,
                sign_key: str | None = None,
                require_signature: bool = False,
                ed25519_pub: str | Path | None = None) -> DeployedScorer:
    """部署侧：加载 .pkg 并重建编码器(必须是打包容器的同一编码器)。"""
    from .encoders import get_encoder

    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    manifest: dict[str, Any] = {}
    digest = ""
    if isinstance(obj, dict) and obj.get("format") == "poisonguard.pkg":
        payload = obj.get("payload") or b""
        want = str(obj.get("sha256") or "")
        got = hashlib.sha256(payload).hexdigest()
        if want and got != want:
            raise ValueError(
                f".pkg 完整性校验失败: sha256 {got[:16]}… != {want[:16]}…"
                "（文件被截断或篡改，拒绝加载）")
        manifest = obj.get("manifest") or {}
        digest = got
        sig = str(obj.get("sig") or "")
        sig_ed = str(obj.get("sig_ed25519") or "")
        if sig_ed:
            if not ed25519_pub:
                raise ValueError(
                    ".pkg 用 ed25519 签名（作者私钥），未提供验证公钥"
                    "（--verify-ed25519 author.pub）")
            obj["_pub"] = _load_ed25519_public(ed25519_pub)
            ok_ed = _verify_container_ed25519(obj, sig_ed)
            obj.pop("_pub", None)
            if not ok_ed:
                raise ValueError(
                    ".pkg ed25519 签名校验失败（公钥不匹配，或包在签名后被改动/替换）")
        elif sig:
            if not sign_key:
                raise ValueError(
                    ".pkg 已用 HMAC 签名，但未提供校验密钥"
                    "（--sign-key / --sign-key-file / 环境变量 POISONGUARD_PKG_KEY）")
            if not hmac.compare_digest(sig, _sign_container(obj, sign_key)):
                raise ValueError(
                    ".pkg 签名校验失败（密钥不对，或包在签名后被改动/替换）")
        elif require_signature:
            raise ValueError(".pkg 未签名，但要求必须签名（--require-signature）；"
                             "请让作者侧用 --sign-key 或 --sign-ed25519 重新打包")
        state: DeployState = pickle.loads(payload)
    else:  # 兼容旧格式：裸 DeployState（无 sha256/manifest）
        state = obj
        print("[run] 旧格式 .pkg（无 sha256/manifest），建议重新 deploy 以启用校验")
    if getattr(state, "version", 0) != PKG_VERSION:
        raise ValueError(f".pkg 版本不兼容: 包内 {getattr(state,'version',0)} vs 运行时 {PKG_VERSION}")

    base = cfg or Config()
    enc_spec = state.encoder or {}
    name = enc_spec.get("name", base.encoder.name)
    if name != base.encoder.name:
        # 换编码器 = 换特征空间，kNN/学生全部失效 → 必须同编码器
        base.encoder.name = name
    if enc_spec.get("max_length"):
        base.encoder.max_length = int(enc_spec["max_length"])
    try:
        enc = get_encoder(base)
    except FileNotFoundError:
        raise FileNotFoundError(
            "部署机缺少编码器权重(公共模型)。请用 `python -m poisonguard.cli "
            "download-encoder --encoder crom` 下载后重试。"
        )
    want_fp = str(manifest.get("encoder_fingerprint") or "")
    if want_fp:
        try:
            got_fp = encoder_fingerprint(base)
        except OSError:
            got_fp = ""
        if got_fp and got_fp != want_fp:
            raise ValueError(
                f"编码器权重指纹不一致: 包内 {want_fp} vs 本地 {got_fp}"
                "（同名的不同权重会让 kNN/学生全部失真，拒绝加载）")
    sc = DeployedScorer(state, enc)
    quant = manifest.get("quantize") or state.provenance.get("quantize", "none")
    thr_src = manifest.get("threshold_source") or "scorer_default"
    print(f"[run] 阈值 {sc.threshold:.4f}（来源 {thr_src}）")
    print(f"[run] 发布物校验通过: domain={manifest.get('domain') or state.domain} "
          f"quantize={quant} "
          f"sha256={(digest[:16] + '…') if digest else '(legacy, 无校验)'} "
          f"签名={'已验(' + str(manifest.get('signed')) + ')' if manifest.get('signed') else '无'} "
          f"指纹={(manifest.get('encoder_fingerprint') or '-')}")
    return sc


# ----------------------------------------------------------------------
def cmd_deploy(cfg: Config, args: argparse.Namespace) -> Path:
    return bundle_scorer(
        cfg, out_path=getattr(args, "out", None),
        student_path=getattr(args, "student", None),
        threshold=getattr(args, "threshold", None),
        calibrate=bool(getattr(args, "calibrate", True)),
        quantize=getattr(args, "quantize", "none") or "none",
        sign_key=_resolve_sign_key(getattr(args, "sign_key", None),
                                   getattr(args, "sign_key_file", None)),
        ed25519_key=getattr(args, "sign_ed25519", None))


def cmd_run(cfg: Config, args: argparse.Namespace) -> dict:
    """部署侧批扫：加载 .pkg → 复用 scan 报告管线（注入 scorer，不重新装配）。"""
    if not getattr(args, "bundle", None):
        print("[run] 需要 --bundle path.pkg")
        import sys
        sys.exit(2)
    from .scan import scan_file

    scorer = load_bundle(
        args.bundle, cfg,
        sign_key=_resolve_sign_key(getattr(args, "sign_key", None),
                                   getattr(args, "sign_key_file", None)),
        require_signature=bool(getattr(args, "require_signature", False)),
        ed25519_pub=getattr(args, "verify_ed25519", None))
    result = scan_file(cfg, args.input, threshold=None,
                       judge_topk=getattr(args, "judge_topk", 0),
                       fmt=getattr(args, "format", "auto"),
                       column=getattr(args, "column", "text"),
                       limit=getattr(args, "limit", 0),
                       render_html=not getattr(args, "no_html", False),
                       out_prefix=getattr(args, "out_prefix", None),
                       scorer=scorer)
    return result


if __name__ == "__main__":
    import sys
    from .config import load_config

    ap = argparse.ArgumentParser(description="poisonguard 部署打包/离线评分")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("deploy", help="把训练产物打成 .pkg")
    p1.add_argument("--config", default=None)
    p1.add_argument("--domain", default="", help="processed 目录后缀(如 afqmc)；空=默认 v3")
    p1.add_argument("--student", default=None)
    p1.add_argument("--threshold", type=float, default=None)
    p1.add_argument("--out", default=None)
    p1.add_argument("--quantize", default="none", choices=["none", "f16", "i8"],
                    help="kNN 参考向量量化(none/f16/i8)，用于压缩发布物体积")
    p1.add_argument("--sign-key", default=None, help="HMAC 签名密钥(也可用环境变量)")
    p1.add_argument("--sign-key-file", default=None, help="从文件读签名密钥")
    p2 = sub.add_parser("run", help="加载 .pkg 离线批扫")
    p2.add_argument("--config", default=None)
    p2.add_argument("--bundle", required=True)
    p2.add_argument("--input", required=True)
    p2.add_argument("--format", default="auto")
    p2.add_argument("--column", default="text")
    p2.add_argument("--limit", type=int, default=0)
    p2.add_argument("--judge-topk", type=int, default=0)
    p2.add_argument("--out-prefix", default=None)
    p2.add_argument("--no-html", action="store_true")
    p2.add_argument("--sign-key", default=None, help="校验 HMAC 签名的密钥")
    p2.add_argument("--sign-key-file", default=None, help="从文件读校验密钥")
    p2.add_argument("--require-signature", action="store_true",
                    help="包未签名则拒绝加载")
    a = ap.parse_args(sys.argv[1:])
    cfg = load_config(a.config)
    if getattr(a, "domain", ""):
        cfg.data.processed_dir = str(
            Path(cfg.data.processed_dir).with_name(f"processed_{a.domain}"))
    (cmd_deploy if a.cmd == "deploy" else cmd_run)(cfg, a)
