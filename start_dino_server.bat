@echo off
REM DINO 서버 자동 시작 스크립트 (Windows)

REM 프로젝트 루트로 이동
cd /d "%~dp0"

REM 서버가 이미 실행 중인지 확인
netstat -ano | findstr :5001 >nul
if %errorlevel% == 0 (
    echo ✅ DINO 서버가 이미 실행 중입니다 (포트 5001)
    exit /b 0
)

REM Python 경로 확인
where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Python을 찾을 수 없습니다
        exit /b 1
    )
    set PYTHON_CMD=python3
) else (
    set PYTHON_CMD=python
)

echo 🚀 DINO 서버 시작 중...
echo    Python: %PYTHON_CMD%
echo    포트: 5001
echo.

REM 모델 경로 확인
if not exist "models\dino\BoltDINO.pt" (
    echo ⚠️  경고: models\dino\BoltDINO.pt 파일을 찾을 수 없습니다
)

if not exist "models\dino\DoorDINO_high.pt" (
    echo ⚠️  경고: models\dino\DoorDINO_high.pt 파일을 찾을 수 없습니다
)

if not exist "models\dino\DoorDINO_mid.pt" (
    echo ⚠️  경고: models\dino\DoorDINO_mid.pt 파일을 찾을 수 없습니다
)

if not exist "models\dino\DoorDINO_low.pt" (
    echo ⚠️  경고: models\dino\DoorDINO_low.pt 파일을 찾을 수 없습니다
)

REM 서버 시작
%PYTHON_CMD% dino_server.py --port 5001 ^
  --bolt-model models/dino/BoltDINO.pt ^
  --door-high-model models/dino/DoorDINO_high.pt ^
  --door-mid-model models/dino/DoorDINO_mid.pt ^
  --door-low-model models/dino/DoorDINO_low.pt

if %errorlevel% neq 0 (
    echo.
    echo ❌ 서버 시작 실패
    echo.
    echo 문제 해결:
    echo 1. Python이 설치되어 있는지 확인: %PYTHON_CMD% --version
    echo 2. 필요한 패키지가 설치되어 있는지 확인: pip install -r requirements.txt
    echo 3. 모델 파일이 올바른 경로에 있는지 확인
    exit /b 1
)

