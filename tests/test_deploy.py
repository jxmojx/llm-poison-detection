"""deploy 打包/离线批扫的回归测试：状态打包 roundtrip / 版本校验 / 评分回落逻辑。

不下载模型、不跑训练，只用假编码器与假拟合对象验证"打包产品逻辑对不对"
（与 test_v3.py / test_scan.py 同哲学）。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poisonguard.deploy import (  # noqa: E402
    PKG_VERSION, DeployState, DeployedScorer, load_bundle,
)
from poisonguard.detectors import Calibration, CharNgramDetector, KNNDetector  # noqa: E402


class _FakeEnc:
    """假编码器：encode -> (n, 4) 全 1 向量，无需真实权重。"""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=float)


class _FakeDet:
    """假拟合对象：score/to_fpr 返回固定值。"""

    def __init__(self, value: float):
        self.value = value

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        return np.full(len(texts), self.value, dtype=float)

    def to_fpr(self, scores: np.ndarray) -> np.ndarray:
        return np.full(len(scores), self.value, dtype=float)


# ----------------------------------------------------------------------
def _state(version: int = PKG_VERSION, **kw) -> DeployState:
    defaults = dict(
        domain="afqmc",
        encoder={"name": "crom", "max_length": 64, "pooling": "mean"},
        detector={"target_fpr": 0.01, "knn_k": 5},
        threshold=0.897,
        judge_threshold=0.8845,
        knn=None, ngram=None, cal=None, student=None,
        provenance={"encoder": "crom", "has_student": True},
    )
    defaults.update(kw)
    return DeployState(version=version, **defaults)


def test_pkg_state_roundtrip(tmp_path: Path):
    """DeployState 可 pickle 往返，关键字段不丢。"""
    s = _state(provenance={"created": "2026-11-11", "encoder": "crom"})
    p = tmp_path / "x.pkg"
    with open(p, "wb") as f:
        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(p, "rb") as f:
        s2 = pickle.load(f)
    assert s2.version == PKG_VERSION
    assert s2.domain == "afqmc"
    assert s2.encoder == {"name": "crom", "max_length": 64, "pooling": "mean"}
    assert s2.threshold == 0.897
    assert s2.judge_threshold == 0.8845
    assert s2.provenance["encoder"] == "crom"


def test_deploy_state_can_carry_fitted_objects(tmp_path: Path):
    """真正的拟合态(纯 sklearn)应能随包 pickle（不要求模型权重）。"""
    knn = KNNDetector(k=5)
    # 用极小参考集 fit，验证 KNNDetector 可序列化
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float)
    knn.fit(X, ["a", "b", "c"])
    cal = Calibration.fit(np.array([0.1, 0.2, 0.3, 0.4]), 0.01)
    ng = CharNgramDetector()
    # 至少两个类别，否则 LogisticRegression 拒绝拟合（真实场景 dev 必含正负类）
    ng.fit_labeled(["正常句一 平安银行", "正常句二 理财", "投毒句 借呗秒批"],
                   np.array([0, 0, 1]))
    s = _state(knn=knn, cal=cal, ngram=ng)
    p = tmp_path / "fit.pkg"
    with open(p, "wb") as f:
        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(p, "rb") as f:
        s2 = pickle.load(f)
    assert s2.knn is not None and s2.cal is not None and s2.ngram is not None
    # 拟合态仍可打分（无需编码器）
    knn_scores = s2.knn.score(X, ["a", "b", "c"])
    assert knn_scores.shape == (3,)
    assert float(np.isfinite(knn_scores).all()) == 1.0


def test_load_bundle_version_mismatch_raises(tmp_path: Path):
    """版本不匹配必须在碰编码器之前就抛 ValueError。"""
    p = tmp_path / "old.pkg"
    with open(p, "wb") as f:
        pickle.dump(_state(version=999), f, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="版本不兼容"):
        load_bundle(p)


def test_deployed_scorer_score_fallback_lanes():
    """DeployedScorer.score 的回落链：student -> 1-fpr -> knn。"""
    # student 有界 → 用 student
    s1 = _state(knn=_FakeDet(0.2), cal=_FakeDet(0.5),
                ngram=_FakeDet(0.9), student=_FakeDet(0.7))
    sc1 = DeployedScorer(s1, _FakeEnc())
    r1 = sc1.score(["甲", "乙"], explain=True)
    assert [x["score"] for x in r1["results"]] == [0.7, 0.7]
    assert r1["results"][0]["views"]["student"] == 0.7

    # student 缺失 → 回落到 1-fpr（fpr=0.5 → 0.5）
    s2 = _state(knn=_FakeDet(0.2), cal=_FakeDet(0.5),
                ngram=_FakeDet(0.9), student=None)
    sc2 = DeployedScorer(s2, _FakeEnc())
    r2 = sc2.score(["丙"], explain=True)
    assert r2["results"][0]["score"] == 0.5

    # 无 cal → 回落到 knn
    s3 = _state(knn=_FakeDet(0.3), cal=None, ngram=None, student=None)
    sc3 = DeployedScorer(s3, _FakeEnc())
    r3 = sc3.score(["丁"], explain=True)
    assert r3["results"][0]["score"] == 0.3

    # flag 按阈值（sc1 的 student=0.7 < 阈值 0.897 → False）
    assert sc1.score(["甲"], explain=False)["results"][0]["flag"] is False
    # 再验证一个超过阈值的
    s4 = _state(knn=_FakeDet(0.3), cal=None, ngram=None, student=None, threshold=0.2)
    sc4 = DeployedScorer(s4, _FakeEnc())
    assert sc4.score(["戊"])["results"][0]["flag"] is True


def test_deployed_scorer_properties():
    sc = DeployedScorer(_state(), _FakeEnc())
    assert sc.version == f"pkg-v{PKG_VERSION}/afqmc"
    assert sc.threshold == 0.897
    assert sc.judge_threshold == 0.8845
    assert sc.student_path == "bundled"
    assert sc.has_student is False  # 本 state student=None
    # 报告口径：领域名/编码器来自包内，不依赖部署机的 processed_<域> 目录
    assert sc.domain == "afqmc"
    assert sc.encoder_name == "crom"
    sc2 = DeployedScorer(_state(student=_FakeDet(0.7)), _FakeEnc())
    assert sc2.has_student is True


def test_domain_tag_strips_leading_underscore():
    """processed_afqmc -> afqmc（否则 judge__afqmc.json 永远找不到，兜底会误取别的域阈值）。"""
    from types import SimpleNamespace

    from poisonguard.deploy import _domain_tag

    def cfg(path: str):
        return SimpleNamespace(data=SimpleNamespace(processed_dir=path))

    assert _domain_tag(cfg("data/processed_afqmc")) == "afqmc"
    assert _domain_tag(cfg("data/processed_real_v3")) == "real_v3"
    assert _domain_tag(cfg("data/processed")) == ""


# ----------------------------------------------------------------------
# 发布物体积与完整性：kNN 量化 / sha256 校验
# ----------------------------------------------------------------------
def test_packed_knn_close_to_full_precision():
    """f16/i8 量化后的 kNN 距离必须贴近全精度（决策层不因此改变）。"""
    from poisonguard.deploy import PackedKNN

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = rng.normal(size=(20, 32)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    knn = KNNDetector(k=5).fit(X, [""] * len(X))
    base = knn.score(Q, [""])
    f16 = PackedKNN.from_knn(knn, mode="f16")
    i8 = PackedKNN.from_knn(knn, mode="i8")
    assert np.abs(f16.score(Q, [""]) - base).max() < 1e-3
    assert np.abs(i8.score(Q, [""]) - base).max() < 1e-2
    # 体积：i8 码字应是 float32 的 1/4
    assert i8.codes.nbytes <= X.nbytes // 4
    # 语义与接口对齐 KNNDetector
    assert f16.name == "knn" and i8.name == "knn"


def _write_container(path, state, *, tamper: bool = False):
    import hashlib

    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    if tamper:
        digest = "0" * 64
    container = {"format": "poisonguard.pkg", "version": PKG_VERSION,
                 "sha256": digest, "bytes": len(payload),
                 "manifest": {"domain": "afqmc", "quantize": "i8"},
                 "payload": payload}
    path.write_bytes(pickle.dumps(container))
    return path


def test_pkg_sha256_tamper_rejected(tmp_path, monkeypatch):
    """payload 与 sha256 不一致时必须拒绝加载（截断/篡改）。"""
    _write_container(tmp_path / "bad.pkg", _state(), tamper=True)
    monkeypatch.setattr("poisonguard.encoders.get_encoder", lambda cfg: _FakeEnc())
    with pytest.raises(ValueError, match="完整性校验失败"):
        load_bundle(tmp_path / "bad.pkg")


def test_pkg_container_roundtrip_verifies(tmp_path, monkeypatch):
    """正常容器：sha256 校验通过，manifest 里的量化口径随包带回。"""
    p = _write_container(tmp_path / "ok.pkg", _state())
    monkeypatch.setattr("poisonguard.encoders.get_encoder", lambda cfg: _FakeEnc())
    sc = load_bundle(p)
    assert sc.domain == "afqmc"
    assert sc.state.provenance  # 拟合态 provenance 随包


# ----------------------------------------------------------------------
# 发布物真实性：HMAC 签名 + 编码器权重指纹
# ----------------------------------------------------------------------
def _write_container2(path: Path, state: DeployState, *, key: str | None = None,
                      manifest: dict | None = None):
    """造容器（可选签名）：sig 覆盖 format|version|sha256|manifest。"""
    import hashlib

    from poisonguard.deploy import _sign_container

    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    container = {
        "format": "poisonguard.pkg",
        "version": PKG_VERSION,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "manifest": manifest if manifest is not None else {"domain": "afqmc"},
        "payload": payload,
    }
    if key:
        container["sig"] = _sign_container(container, key)
    path.write_bytes(pickle.dumps(container))
    return path


@pytest.fixture()
def fake_encode(monkeypatch):
    """不打真模型：编码器换成假实现，指纹计算短路为不参与校验。"""
    monkeypatch.setattr("poisonguard.encoders.get_encoder", lambda cfg: _FakeEnc())
    monkeypatch.setattr("poisonguard.deploy.encoder_fingerprint", lambda cfg: "")
    return True


def test_pkg_hmac_signature_roundtrip(tmp_path: Path, fake_encode):
    """密钥对 → 通过；密钥错 → 签名失败；没给密钥 → 拒绝（不能静默跳过校验）。"""
    p = _write_container2(tmp_path / "signed.pkg", _state(), key="k-secret")
    sc = load_bundle(p, sign_key="k-secret")
    assert sc.domain == "afqmc"
    with pytest.raises(ValueError, match="签名校验失败"):
        load_bundle(p, sign_key="wrong-key")
    with pytest.raises(ValueError, match="未提供校验密钥"):
        load_bundle(p)


def test_pkg_signed_container_binds_manifest_and_payload(tmp_path: Path, fake_encode):
    """改 manifest（如把阈值调低）或换 payload，签名都必须失效。"""
    p = _write_container2(tmp_path / "s.pkg", _state(), key="k",
                          manifest={"domain": "afqmc", "threshold": 0.9})
    pkg = pickle.loads(p.read_bytes())
    pkg["manifest"]["threshold"] = 0.1        # 只改 manifest
    p.write_bytes(pickle.dumps(pkg))
    with pytest.raises(ValueError, match="签名校验失败"):
        load_bundle(p, sign_key="k")


def test_pkg_require_signature_rejects_unsigned(tmp_path: Path, fake_encode):
    p = _write_container2(tmp_path / "plain.pkg", _state(), key=None)
    load_bundle(p)                                    # 默认允许未签名
    with pytest.raises(ValueError, match="未签名"):
        load_bundle(p, require_signature=True)


def test_load_bundle_rejects_encoder_fingerprint_mismatch(tmp_path, monkeypatch):
    """同名不同权重（指纹不一致）必须拒绝：否则 kNN/学生特征空间是错的。"""
    monkeypatch.setattr("poisonguard.encoders.get_encoder", lambda cfg: _FakeEnc())
    monkeypatch.setattr("poisonguard.deploy.encoder_fingerprint", lambda cfg: "cafebabe0123")
    p = _write_container2(tmp_path / "fp.pkg", _state(),
                          manifest={"domain": "afqmc",
                                    "encoder_fingerprint": "deadbeef9999"})
    with pytest.raises(ValueError, match="指纹不一致"):
        load_bundle(p)


def test_encoder_fingerprint_tracks_weight_bytes(tmp_path: Path):
    """指纹 = 文件名+大小+内容；权重变一个字节就要变，路径/内容相同则稳定。"""
    from types import SimpleNamespace

    from poisonguard.deploy import encoder_fingerprint

    d = tmp_path / "model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x" * 2048)
    (d / "config.json").write_bytes(b"{}")           # 非权重文件不参与
    cfg = SimpleNamespace(encoder=SimpleNamespace(model_dir=str(d)))

    fp1 = encoder_fingerprint(cfg)
    fp2 = encoder_fingerprint(cfg)
    assert fp1 and fp1 == fp2 and len(fp1) == 16

    (d / "model.safetensors").write_bytes(b"x" * 2047 + b"y")   # 同大小、改内容
    assert encoder_fingerprint(cfg) != fp1

    (d / "model.safetensors").write_bytes(b"x" * 2048)          # 还原
    assert encoder_fingerprint(cfg) == fp1
