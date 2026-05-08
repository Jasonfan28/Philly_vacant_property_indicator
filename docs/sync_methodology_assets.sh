#!/usr/bin/env bash
# Copy methodology figure assets from the PhillyStat360 R project into the website
# so PhillyStat360 v2.html can resolve its <img> paths.
#
# Run from the website root:  bash sync_methodology_assets.sh
# Re-run any time the source figures regenerate.

set -euo pipefail

SRC="${PHILLYSTAT_SRC:-G:/My Drive/PhillyStat_R/PhillyStat360}"
DST="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  echo "Set PHILLYSTAT_SRC to the PhillyStat360 R project root." >&2
  exit 1
fi

copy_dir() {
  local rel="$1"
  local src_path="$SRC/$rel"
  local dst_path="$DST/$rel"
  if [[ ! -d "$src_path" ]]; then
    echo "  skip (missing): $rel"
    return
  fi
  mkdir -p "$(dirname "$dst_path")"
  cp -r "$src_path" "$(dirname "$dst_path")/"
  echo "  ok: $rel"
}

echo "Syncing methodology assets"
echo "  from: $SRC"
echo "  to:   $DST"

# Remove any prior copies so renames in the source don't leave stragglers
rm -rf "$DST/graphs" \
       "$DST/code/outputs" \
       "$DST/code/03_1_Ovs_files" \
       "$DST/code/03_2_Analysis_files"

copy_dir "graphs"
copy_dir "code/outputs"
copy_dir "code/03_1_Ovs_files"
copy_dir "code/03_2_Analysis_files"

echo "Done."
