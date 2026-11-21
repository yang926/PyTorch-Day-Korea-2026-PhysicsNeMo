#!/bin/bash
set -euo pipefail
ptday_ref="${PTDAY_COURSE_REF:-main}"
if [[ "$ptday_ref" != "main" && ! "$ptday_ref" =~ ^[a-f0-9]{40}$ ]]; then
  echo 'PTDAY_COURSE_REF must be main or a full lowercase commit SHA.' >&2
  exit 2
fi
ptday_bootstrap="$(mktemp -t ptday-bootstrap.XXXXXX)"
curl --fail --silent --show-error --location --retry 3 \
  --connect-timeout 20 --max-time 120 \
  "https://raw.githubusercontent.com/yang926/PyTorch-Day-Korea-2026-PhysicsNeMo/$ptday_ref/brev/bootstrap.py" \
  --output "$ptday_bootstrap"
python3 "$ptday_bootstrap" --ref "$ptday_ref"
