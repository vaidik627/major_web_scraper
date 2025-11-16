#!/bin/bash

# AI Web Scraper Setup Script
echo "🚀 Setting up AI Web Scraper..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment files
echo "📝 Creating environment files..."

if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env file and add your API keys"
fi

if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "✅ Created backend/.env file from template"
fi

if [ ! -f frontend/.env ]; then
    cp frontend/.env.example frontend/.env
    echo "✅ Created frontend/.env file from template"
fi

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📱 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo ""
    echo "🔑 Don't forget to:"
    echo "   1. Add your OpenAI API key to .env file"
    echo "   2. Add your Anthropic API key to .env file (optional)"
    echo "   3. Restart services: docker-compose restart"
    echo ""
    echo "📖 For more information, check the README.md file"
else
    echo "❌ Some services failed to start. Check logs with: docker-compose logs"
fi