@echo off
REM Push this folder to a (re)created GitHub repo.
REM
REM BEFORE running: create the new empty repo on github.com
REM   -> https://github.com/new
REM   -> do NOT add a README/.gitignore/license (keep it empty)
REM   -> copy its HTTPS URL, e.g. https://github.com/xylinum97/SCF_Research_RL_Imperial.git

setlocal enabledelayedexpansion

set /p REPOURL="https://github.com/xylinum97/SCF_Research_RL_Imperial.git"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This folder is not a git repository. Aborting.
    exit /b 1
)

REM Repoint 'origin' at the new repo (old one was deleted)
git remote remove origin >nul 2>&1
git remote add origin "!REPOURL!"

echo.
echo Current status:
git status

REM Stage everything not covered by .gitignore
git add -A

git diff --cached --quiet
if not errorlevel 1 (
    echo Nothing staged to commit.
) else (
    set /p COMMITMSG="Commit message: "
    git commit -m "!COMMITMSG!"
)

git branch -M main
git push -u origin main

echo Done.
endlocal
