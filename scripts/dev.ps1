$ErrorActionPreference = "Stop"
$python = "C:\Users\Ninad Naik\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
Start-Process -WindowStyle Hidden -FilePath $python -ArgumentList "-m","uvicorn","api.index:app","--host","127.0.0.1","--port","8000"
npm run dev
