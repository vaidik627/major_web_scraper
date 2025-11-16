# 🔐 Environment Variables Setup Guide

## Overview
Your project uses environment variables to securely manage sensitive data like API keys. These are **never committed to Git** and remain private on your machine.

## Quick Setup

### 1. **Backend Setup** (`.env` in `backend/` folder)

Copy `.env.example` to `.env` and fill in your values:

```bash
# Database
DATABASE_URL=sqlite:///./scraper.db

# AI APIs
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your_super_secret_key_here_change_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Scraping
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36

# App Configuration
APP_NAME=AI Web Scraper

# Frontend URL (for email links)
FRONTEND_URL=http://localhost:3000

# Email Configuration (SendGrid)
SENDGRID_API_KEY=SG.your_actual_sendgrid_api_key_here
SENDER_EMAIL=your_verified_email@gmail.com
SENDER_NAME=Your Name - AI Web Scraper
```

### 2. **Frontend Setup** (`.env` in `frontend/` folder)

Copy `frontend/.env.example` to `frontend/.env`:

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000
```

## 🔑 Getting Your SendGrid API Key

1. Go to [SendGrid Dashboard](https://app.sendgrid.com)
2. Navigate to **Settings → API Keys**
3. Click **Create API Key**
4. Give it a name (e.g., "AI Web Scraper")
5. Copy the key (starts with `SG.`)
6. Paste it in your `.env` file as `SENDGRID_API_KEY=SG.xxxxx`

## ✅ Security Best Practices

- ✅ **Never commit `.env` files** - They're in `.gitignore`
- ✅ **Use `.env.example`** - Share this with team, not `.env`
- ✅ **Different keys per environment** - Dev, staging, production
- ✅ **Rotate keys regularly** - Especially if exposed
- ✅ **Use strong secret keys** - For `SECRET_KEY` in production

## 🚀 For Production Deployment

When deploying to platforms like **Heroku**, **Vercel**, or **AWS**:

1. Set environment variables in the platform's dashboard
2. **Never** paste `.env` file content
3. Use the platform's secrets management

### Example for Heroku:
```bash
heroku config:set SENDGRID_API_KEY=SG.your_key_here
heroku config:set SENDER_EMAIL=your_email@gmail.com
heroku config:set SECRET_KEY=your_production_secret_key
```

## 📝 Troubleshooting

**Emails not sending?**
- Check `SENDGRID_API_KEY` is set correctly
- Verify sender email is verified in SendGrid
- Check backend logs for errors

**App won't start?**
- Ensure `.env` file exists in `backend/` folder
- Check for typos in variable names
- Verify all required variables are set

## 📚 Related Files
- `backend/.env.example` - Backend template
- `frontend/.env.example` - Frontend template
- `.gitignore` - Ensures `.env` files aren't committed
