@echo off
title WorldQuant Brain Genius Multi-Region Miner
echo Starting Genius Alpha Miner...
cd /d "C:\Users\HP\H_PROJECT_RESEARCH_PACKAGE_CLEAN"
"C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" gp_genius_miner.py %1
echo.
echo Miner has stopped. Press any key to exit.
pause > nul
