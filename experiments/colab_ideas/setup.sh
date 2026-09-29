#!/usr/bin/env bash
# Colab setup for the paraphrase-attack experiments. Run in the background from a notebook cell:
#   import subprocess; subprocess.Popen(["bash", "/content/setup.sh"], stdout=open("/content/setup.log", "w"), stderr=subprocess.STDOUT)
# Takes ~3-5 min; ends with SETUP_DONE in /content/setup.log. Afterwards restart Ollama with the
# tuned settings from Python: import extras; extras.start_ollama()
# MODELS overrides the models to pull (follow-up F needs only llama3.1:8b, gemma3:12b and
# qwen3:4b-instruct-2507-q4_K_M). OLLAMA_VERSION defaults to the version of the 2026-09-26 run.
set -x
apt-get -qq install -y zstd pciutils > /dev/null 2>&1
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=${OLLAMA_VERSION:-0.34.4} sh
OLLAMA_HOST=127.0.0.1:11434 nohup ollama serve > /tmp/ollama_setup.log 2>&1 &
for i in $(seq 60); do curl -s 127.0.0.1:11434/api/tags > /dev/null && break; sleep 1; done
ollama --version
# Guy's branch, pinned to the commit the 2026-09-26 run used (a1e21e5 = tip of minimal-edit-paraphrasing then)
rm -rf /content/avp && git clone -q --branch minimal-edit-paraphrasing https://github.com/GuyZilberman/AttackingViaParaphrasing /content/avp
git -C /content/avp checkout -q a1e21e5 && git -C /content/avp log --oneline -1
# Pull everything before any inference: pulling while models run caused disk/RAM contention
for m in ${MODELS:-qwen3:4b llama3.1:8b gemma3:12b qwen3:4b-instruct-2507-q4_K_M mistral-nemo:12b}; do
  ollama pull $m > /tmp/pull.log 2>&1; echo "pulled $m: $?"
done
ollama list
echo SETUP_DONE
