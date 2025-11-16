@echo off
echo 🚀 Setting up AI Web Scraper...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create environment files
echo 📝 Creating environment files...

if not exist .env (
    copy .env.example .env
    echo ✅ Created .env file from template
    echo ⚠️  Please edit .env file and add your API keys
)

if not exist backend\.env (
    copy backend\.env.example backend\.env
    echo ✅ Created backend\.env file from template
)

if not exist frontend\.env (
    copy frontend\.env.example frontend\.env
    echo ✅ Created frontend\.env file from template
)

REM Build and start services
echo 🐳 Building and starting Docker containers...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check if services are running
docker-compose ps | findstr "Up" >nul
if %errorlevel% equ 0 (
    echo ✅ Services are running!
    echo.
    echo 🎉 Setup complete!
    echo.
    echo 📱 Frontend: http://localhost:3000
    echo 🔧 Backend API: http://localhost:8000
    echo 📚 API Documentation: http://localhost:8000/docs
    echo.
    echo 🔑 Don't forget to:
    echo    1. Add your OpenAI API key to .env file
    echo    2. Add your Anthropic API key to .env file (optional)
    echo    3. Restart services: docker-compose restart
    echo.
    echo 📖 For more information, check the README.md file
) else (
    echo ❌ Some services failed to start. Check logs with: docker-compose logs
)

pause