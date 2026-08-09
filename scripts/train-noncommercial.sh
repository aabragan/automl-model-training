#!/usr/bin/env bash
# Train with the AutoGluon `noncommercial` preset (extreme + TabPFN-3).
#
# TabPFN-3 requires a Prior Labs API key. This script refuses to run unless
# a `tabpfn.env` file exists in the repo root containing:
#
#   TABPFN_TOKEN=<your Prior Labs API key>
#
# Get a key: log in at https://ux.priorlabs.ai/account, accept the license
# under Account -> Licenses, and copy the API key. tabpfn.env is git-ignored;
# never commit it.
#
# LICENSING: TabPFN-3 is free for research and internal experimentation only.
# Commercial use requires a license from Prior Labs:
# https://docs.priorlabs.ai/models#tabpfn-model-license
#
# Usage:
#   ./scripts/train-noncommercial.sh <data.csv> [train flags...]
#
# All arguments are forwarded to `uv run train`. The preset is always
# `noncommercial`; passing your own --preset is rejected.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/tabpfn.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found." >&2
    echo "" >&2
    echo "The noncommercial preset uses TabPFN-3, which needs a Prior Labs API key." >&2
    echo "Create tabpfn.env in the repo root containing:" >&2
    echo "" >&2
    echo "  TABPFN_TOKEN=<your Prior Labs API key>" >&2
    echo "" >&2
    echo "Get a key at https://ux.priorlabs.ai/account (accept the license under" >&2
    echo "Account -> Licenses). The file is git-ignored; never commit it." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${TABPFN_TOKEN:-}" ]]; then
    echo "ERROR: $ENV_FILE exists but does not set TABPFN_TOKEN." >&2
    echo "Expected a line of the form: TABPFN_TOKEN=<your Prior Labs API key>" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <data.csv> [train flags...]" >&2
    echo "Example: $0 data.csv --label target --time-limit 3600" >&2
    exit 1
fi

for arg in "$@"; do
    if [[ "$arg" == "--preset" ]]; then
        echo "ERROR: do not pass --preset; this script always trains with --preset noncommercial." >&2
        exit 1
    fi
done

cd "$REPO_ROOT"
exec uv run train "$@" --preset noncommercial
