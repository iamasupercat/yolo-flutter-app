#!/bin/bash
# DINO 서버 자동 시작 스크립트

# 프로젝트 루트로 이동
cd "$(dirname "$0")"

# 서버가 이미 실행 중인지 확인
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ DINO 서버가 이미 실행 중입니다 (포트 5001)"
    exit 0
fi

# Python 경로 확인
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python3를 찾을 수 없습니다"
    exit 1
fi

echo "🚀 DINO 서버 시작 중..."
echo "  Python: $PYTHON_CMD"
echo "  포트: 5001"
echo ""

# 서버 실행 (백그라운드)
nohup "$PYTHON_CMD" dino_server.py \
  --port 5001 \
  --bolt-model models/dino/BoltDINO.pt \
  --door-high-model models/dino/DoorDINO_high.pt \
  --door-mid-model models/dino/DoorDINO_mid.pt \
  --door-low-model models/dino/DoorDINO_low.pt \
  > dino_server.log 2>&1 &

SERVER_PID=$!
echo "✅ DINO 서버가 시작되었습니다 (PID: $SERVER_PID)"
echo "  로그 파일: dino_server.log"
echo "  서버 중지: kill $SERVER_PID"
echo ""

# 서버가 시작될 때까지 대기
sleep 3

# 서버 상태 확인
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ 서버가 정상적으로 실행 중입니다"
    exit 0
else
    echo "⚠️  서버 시작 실패. 로그를 확인하세요: dino_server.log"
    exit 1
fi

