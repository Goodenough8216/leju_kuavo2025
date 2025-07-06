#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

cd "$SCRIPT_DIR" || exit

docker build -t kuavo_mpc_wbc_img:0.5 .
