
:: -------------------------------------------------------------

:: set pyv=%UserProfile%\AppData\Local\Programs\Python\Python37\
::set pyv=%UserProfile%\AppData\Local\Programs\Python\Python39\
set pi=%pyv%\Scripts\pip
:: set py=%pyv%\python.exe
set pyv=%~dp0\venv\Scripts
set py=%pyv%\python.exe

set AWS_PROFILE=some-env

set exe="%~dp0\llm_provider.py"

:: %py% -V && ( pause ) || ( pause )
%py% %exe% && ( pause ) || ( pause )
:: %pi% install --upgrade %pkg% && ( pause ) || ( pause )

