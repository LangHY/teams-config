#!/bin/bash
# OpenCode交互模式（基于run命令）
set -e

mkdir -p ~/.config/opencode
cat > ~/.config/opencode/config.json << 'OCEOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen3.6-27B",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "***"
      },
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen3.6-27B",
          "limit": {
            "context": 204800,
            "output": 8192
          }
        }
      }
    }
  }
}
OCEOF

rm -f /dev/lang/offline/workspace/opencode.json /dev/lang/offline/workspace/.opencode/config.json
cd /dev/lang/offline/workspace

echo "OpenCode 交互模式 (run模式)"
echo "输入问题后回车，输入 quit 退出"
echo "=========================="

while true; do
  echo -n "> "
  read -r input
  if [ "$input" = "quit" ] || [ "$input" = "exit" ]; then
    break
  fi
  if [ -n "$input" ]; then
    /dev/lang/offline/hf_offline/opencode run "$input"
  fi
done
