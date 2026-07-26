# Mirage implementation notes

Mirage is implemented as a Vite/React application with a FastAPI-compatible Python
backend in `api/`. The production deployment uses Vercel's Python runtime and the
frontend calls same-origin `/api/v1/*` routes. The local development server proxies
those calls to Uvicorn.

The product defaults to an honest cached-demonstration provider. Every number shown
is derived from the returned token/sample data; no benchmark performance claim is
pre-seeded. Optional Hugging Face and Groq providers are configuration hooks and are
never required for the recruiter demo.

Implementation order: mathematical services and API, interactive UI, persistence and
benchmark workflow, tests, browser QA, then documentation and deployment.
