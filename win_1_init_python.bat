::
:: you might want to check the python dir
::
:: set pyv=%UserProfile%\AppData\Local\Programs\Python\Python37\
set pyv=%UserProfile%\AppData\Local\Programs\Python\Python39\

set py=%pyv%\python.exe
set name=venv

:: %py% -V && ( pause ) || ( pause )
%py% -m venv %name% && ( pause ) || ( pause )
