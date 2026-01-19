@echo off
REM DINO 서버 종료 스크립트 (Windows)

echo DINO 서버 종료 중...

REM 포트 5001을 사용하는 프로세스 찾기 및 종료
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5001 ^| findstr LISTENING') do (
    echo 프로세스 종료 중: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ 서버 종료 완료
    ) else (
        echo ⚠️  프로세스 종료 실패 (이미 종료되었거나 권한이 없을 수 있습니다)
    )
)

REM dino_server.py 프로세스 직접 종료
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *dino_server*" >nul 2>&1
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq *dino_server*" >nul 2>&1

REM 프로세스 이름으로 종료 (더 안전한 방법)
wmic process where "commandline like '%%dino_server.py%%'" delete >nul 2>&1

echo.
echo 서버 종료 완료

