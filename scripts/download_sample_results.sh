#!/usr/bin/env bash
set -euo pipefail

PARENT_FOLDER_ID="1Jen2lklrga45e-h1S_sItHr-_R9qXnhp?usp=sharing"
OUTPUT_DIR="./sample_results"

if ! command -v gdown &> /dev/null; then
  echo "Error: 'gdown' command not found." >&2
  echo "  Install it with: pip install gdown" >&2
  exit 1
fi

PARENT_URL="https://drive.google.com/drive/folders/${PARENT_FOLDER_ID}"

mkdir -p "${OUTPUT_DIR}"

echo "1) Downloading Google Drive folder → ${OUTPUT_DIR}"
gdown "${PARENT_URL}" --folder --remaining-ok --no-cookies -O "${OUTPUT_DIR}"

echo
echo "2) Unzipping all .zip files and removing the originals"
find "${OUTPUT_DIR}" -type f -name '*.zip' | while read -r zip; do
  target_dir=$(dirname "${zip}")
  echo "→ Extracting ${zip} to ${target_dir}/"
  unzip -o "${zip}" -d "${target_dir}"
  echo "→ Deleting ${zip}"
  rm "${zip}"
done

echo
echo "✅ All ZIP archives have been extracted and original .zip files removed."









