# 常用流程（Windows 上装了 make 才能用；没有就直接复制命令）
PY ?= python

.PHONY: help data detect judge distill struct-train audit hardneg report all test serve doctor clean

help:
	@echo "make data     构建数据切分(含 edit/hardneg)"
	@echo "make detect   数据层主实验"
	@echo "make judge    语义层(Qwen 判官, 慢)"
	@echo "make distill  判官蒸馏"
	@echo "make struct-train  结构隔离判官(冻结主干+二分类头, ~40 分钟)"
	@echo "make serve    起在线打分服务"
	@echo "make test     单元测试"
	@echo "make doctor   环境自检"

# v3 口径：真实新闻干净侧 + 内容级投毒 + 硬负样本进主评测集
V3 := --clean-source tnews --edit --hardneg 400 --run-tag v3

data:
	$(PY) -m poisonguard.cli data $(V3)

detect:
	$(PY) -m poisonguard.cli detect $(V3) --boot 300

audit:
	$(PY) -m poisonguard.cli audit $(V3)

hardneg:
	$(PY) -m poisonguard.cli hardneg $(V3)

judge:
	$(PY) -m poisonguard.cli judge $(V3) --qwen-model 0.5b --judge-scoring logprob --limit 40 --dev-limit 20

distill:
	$(PY) -m poisonguard.cli distill $(V3)

struct-train:
	$(PY) -m poisonguard.cli struct-train $(V3) --qwen-model 0.5b

redteam:
	$(PY) -m poisonguard.cli redteam $(V3) --rt-each 5

report:
	$(PY) -m poisonguard.cli report $(V3)

serve:
	$(PY) -m poisonguard.cli serve --port 8000

test:
	$(PY) -m pytest tests/ -q

doctor:
	$(PY) -m poisonguard.cli doctor
