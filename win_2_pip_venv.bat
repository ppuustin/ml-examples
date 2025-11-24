set name=venv
set penv=%~dp0\%name%

set pyt="%penv%\Scripts\python.exe"
set pip="%penv%\Scripts\pip.exe"

set pkg=matplotlib

set mname=fi_core_news_sm
set vers=3.5.0
set model=https://github.com/explosion/spacy-models/releases/download/%mname%-%vers%/%mname%-%vers%-py3-none-any.whl
:: %pyt% -c "import spacy;spacy.load('%mname%').to_disk('src/input/%mname%')" && ( pause ) || ( pause )

:: %pip% show %pkg% && ( pause ) || ( pause )
:: %pyt% -V  && ( pause ) || ( pause )

%pip% install -r "%~dp0\requirements.txt" && ( pause ) || ( pause )
:: %pip% install --no-deps %model%
:: %pip% uninstall %pkg%



:: %pip% install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && ( pause ) || ( pause )
