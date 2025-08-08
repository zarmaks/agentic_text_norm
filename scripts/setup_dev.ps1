param()

Write-Host "Setting up dev environment..." -ForegroundColor Cyan

python --version
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found"; exit 1 }

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import spacy, sys; 
try:
  spacy.load('en_core_web_sm')
  print('spaCy model OK')
except:
  import subprocess; subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])"

Copy-Item .env.example .env -ErrorAction SilentlyContinue

Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "  1) Open .env and add your OPENAI_API_KEY"
Write-Host "  2) Run: .\\.venv\\Scripts\\python.exe scripts\\quick_start.py 'Sony Music/John Doe'"
Write-Host "  3) Run tests: .\\.venv\\Scripts\\python.exe -m pytest"
