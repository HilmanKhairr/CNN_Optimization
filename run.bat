@echo off
title Aplikasi Web Klasifikasi Objek AI (CIFAR-10)
echo ==========================================================
echo   Menjalankan Aplikasi Web secara Lokal...
echo ==========================================================
cd /d "%~dp0"
"%LOCALAPPDATA%\Programs\Python\Python311\Scripts\streamlit.exe" run app.py
pause
