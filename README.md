# Jewellery Catalog App

Ultra-minimal, SEO-first jewellery catalog with single-file Flask backend, PostgreSQL, Docker, and Gemini AI integration.

## Setup

1. Copy `.env.example` to `.env` and add your Gemini API key.
2. Run `docker-compose up --build`
3. Access at `http://localhost:5000`

## Project Hierarchy
```
jewellery-catalog/
├── app.py # Single-file backend (Flask + SQLAlchemy + Gemini)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── templates/
│ ├── base.html
│ ├── index.html
│ ├── product.html
│ ├── search.html
│ └── admin/
│ ├── dashboard.html
│ ├── product_form.html
│ └── categories.html
└── static/
└── uploads/ # User-uploaded images (created at runtime)

```
## Features

- SEO-optimised server-side rendering with clean slugs and JSON‑LD
- Admin panel for product/category management
- AI-powered SEO generation and natural language admin actions (Gemini)
- Image upload with compression and lazy loading
- Full‑text search across product fields and tags
- PostgreSQL persistence with Docker volumes

## Default Categories

Rings, Necklaces, Earrings, Bracelets, Bangles, Chains

## Environment Variables

| Variable          | Description                 |
|-------------------|-----------------------------|
| `SECRET_KEY`      | Flask secret key            |
| `POSTGRES_USER`   | PostgreSQL user             |
| `POSTGRES_PASSWORD`| PostgreSQL password         |
| `POSTGRES_DB`     | PostgreSQL database name    |
| `GEMINI_API_KEY`  | Google Gemini API key       |

## Notes

- Backend is strictly one file (`app.py`)
- No unnecessary abstractions
- Docker volumes persist database and uploaded images