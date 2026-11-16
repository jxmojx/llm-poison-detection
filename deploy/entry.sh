#!/usr/bin/env bash
# poisonguard serve 多副本统一启动入口（配合 docs/多实例部署.md）
# 用法（Compose / k8s 里都一样）：设置好环境变量后执行本脚本。
#
#   SERVE_API_KEY         调用方 Key（必填，开了鉴权后）
#   ADMIN_API_KEY         管理 Key（只给 admin-node 配；LB 副本不配）
#   POISONGUARD_CONFIG    配置文件路径（默认 configs/default.yaml）
#   POISONGUARD_STUDENT   学生模型路径（默认 _default_student 自动找）
#   AUDIT_PATH            审计 sqlite 路径（每副本独立卷）

set -euo pipefail

: "${POISONGUARD_CONFIG:=configs/default.yaml}"
AUDIT_PATH="${AUDIT_PATH:-/tmp/audit/audit.sqlite3}"

ARGS=(serve --host 0.0.0.0 --port 8000 --config "$POISONGUARD_CONFIG")
[ -n "${POISONGUARD_STUDENT:-}" ] && ARGS+=(--student "$POISONGUARD_STUDENT")
[ -n "${SERVE_API_KEY:-}" ] && ARGS+=(--api-key "$SERVE_API_KEY")
[ -n "${ADMIN_API_KEY:-}" ] && ARGS+=(--admin-key "$ADMIN_API_KEY")
ARGS+=(--audit "$AUDIT_PATH")

exec python -m poisonguard.cli "${ARGS[@]}"
