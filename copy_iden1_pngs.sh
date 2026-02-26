#!/usr/bin/env bash
set -euo pipefail

# Copy all files ending with "_iden1.png" into a destination folder.
# Preserves directory structure under the destination to avoid overwriting files
# with the same basename coming from different folders.
#
# Usage:
#   ./copy_iden1_pngs.sh [SRC_ROOT] [DEST_DIR]
# Defaults:
#   SRC_ROOT = .
#   DEST_DIR = res_iden_1

SRC_ROOT="${1:-.}"
DEST_DIR="${2:-res_iden_1}"

# Convert SRC_ROOT to an absolute path (portable on macOS).
SRC_ROOT_ABS="$(cd "$SRC_ROOT" && pwd)"

mkdir -p "$DEST_DIR"

copied=0

# Find all matching files under SRC_ROOT and copy them.
while IFS= read -r -d '' file_path; do
  rel_path="${file_path#"$SRC_ROOT_ABS"/}"
  dest_path="$DEST_DIR/$rel_path"
  mkdir -p "$(dirname "$dest_path")"
  cp -p "$file_path" "$dest_path"
  copied=$((copied + 1))
done < <(find "$SRC_ROOT_ABS" -type f -name '*_iden1.png' -print0)

echo "Copied $copied file(s) into: $(cd "$DEST_DIR" && pwd)"