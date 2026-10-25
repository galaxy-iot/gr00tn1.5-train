#!/usr/bin/env bash
# Download the NVIDIA source revision required by the Qualcomm grootn15 recipe.
set -euo pipefail

revision='4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'
repo_url='https://gh-proxy.org/https://github.com/NVIDIA/Isaac-GR00T.git'
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ${1:-} == '--help' || ${1:-} == '-h' ]]; then
    echo "Usage: bash $0 [destination]"
    echo "Default: $project_root/third_party/Isaac-GR00T"
    echo 'Downloads source only; does not install dependencies or model weights.'
    exit 0
fi
if [[ $# -gt 1 ]]; then
    echo 'Expected at most one destination directory.' >&2
    exit 2
fi
command -v git >/dev/null 2>&1 || { echo 'Please install git first.' >&2; exit 1; }
destination="${1:-$project_root/third_party/Isaac-GR00T}"

if [[ -e "$destination" ]]; then
    if [[ ! -e "$destination/.git" ]]; then
        echo "Destination exists but is not a Git checkout: $destination" >&2
        echo 'Choose a new destination; existing files will not be overwritten.' >&2
        exit 1
    fi
    if [[ "$(git -C "$destination" remote get-url origin)" != "$repo_url" ]]; then
        echo 'The existing origin is not the expected NVIDIA mirror URL. Stopping.' >&2
        exit 1
    fi
    if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=all)" ]]; then
        echo 'The checkout contains local changes or untracked files. Stopping.' >&2
        exit 1
    fi
else
    mkdir -p -- "$destination"
    git -C "$destination" init
    git -C "$destination" remote add origin "$repo_url"
fi

if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
    echo "Downloading NVIDIA source revision $revision ..."
    if ! git -C "$destination" fetch --depth=1 origin "$revision"; then
        echo 'Download failed. Check gh-proxy.org connectivity and rerun this script.' >&2
        echo 'The initialized checkout is retained for retry.' >&2
        exit 1
    fi
fi
git -C "$destination" -c advice.detachedHead=false checkout --detach "$revision"
[[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]]
[[ -f "$destination/pyproject.toml" && -f "$destination/scripts/gr00t_finetune.py" ]]
echo "Source ready: $(cd -- "$destination" && pwd)"
echo "Revision: $revision"
echo 'Activate gr00t-train, then install the dependencies as described in TRAIN_EXPORT.md.'
