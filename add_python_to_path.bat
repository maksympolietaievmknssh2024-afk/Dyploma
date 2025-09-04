@echo off
echo Adding Python 3.10 to PATH environment variable...

:: Add Python and Scripts directories to PATH
setx PATH "%PATH%;%USERPROFILE%\AppData\Local\Programs\Python\Python310;%USERPROFILE%\AppData\Local\Programs\Python\Python310\Scripts"

echo.
echo Python has been added to PATH. Please restart your command prompt for changes to take effect.
echo.
pause