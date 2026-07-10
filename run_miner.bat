@echo off
title WorldQuant Brain GP Alpha Miner
echo Starting Autonomous Alpha Miner...
cd /d "C:\Users\HP\H_PROJECT_RESEARCH_PACKAGE_CLEAN"
"C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" gp_autonomous_miner.py
echo.
echo Miner has stopped. Press any key to exit.
pause > nul
