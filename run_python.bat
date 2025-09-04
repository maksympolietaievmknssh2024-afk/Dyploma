@echo off

:: This batch file makes it easier to run Python commands
:: without typing the full path each time

@echo Running Python command: %*
"%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" %*