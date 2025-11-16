#!/usr/bin/env python3
"""
Debug script for login functionality
Tests the authentication system step by step
"""

import requests
import json
import sys
import os
from datetime import datetime

# Set environment variables
os.environ['SECRET_KEY'] = 'your-super-secret-key-change-in-production-12345'
os.environ['DATABASE_URL'] = 'sqlite:///./scraper.db'
os.environ['ALLOWED_ORIGINS'] = 'http://localhost:3000,http://127.0.0.1:3000'

BASE_URL = "http://localhost:8000/api"

def test_server_connection():
    """Test if the server is running"""
    print("🔍 Testing server connection...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server connection failed: {e}")
        return False

def test_auth_endpoints():
    """Test authentication endpoints"""
    print("\n🔍 Testing authentication endpoints...")
    
    # Test register endpoint
    print("Testing /auth/register endpoint...")
    try:
        register_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }
        response = requests.post(f"{BASE_URL}/auth/register", json=register_data, timeout=10)
        print(f"Register response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Registration successful")
            user_data = response.json()
            print(f"User created: {user_data}")
        elif response.status_code == 400:
            print("⚠️ User might already exist (this is expected if running multiple times)")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Registration failed: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Registration request failed: {e}")

    # Test login endpoint
    print("\nTesting /auth/login endpoint...")
    try:
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=10)
        print(f"Login response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Login successful")
            token_data = response.json()
            print(f"Token received: {token_data}")
            return token_data.get('access_token')
        else:
            print(f"❌ Login failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Login request failed: {e}")
        return None

def test_protected_endpoints(token):
    """Test protected endpoints with token"""
    if not token:
        print("\n⚠️ No token available, skipping protected endpoint tests")
        return
    
    print(f"\n🔍 Testing protected endpoints with token...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test /auth/me endpoint
    print("Testing /auth/me endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers, timeout=10)
        print(f"Me endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ User info retrieved successfully")
            user_info = response.json()
            print(f"User info: {user_info}")
        else:
            print(f"❌ User info retrieval failed: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ User info request failed: {e}")
    
    # Test /auth/verify-token endpoint
    print("\nTesting /auth/verify-token endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/auth/verify-token", headers=headers, timeout=10)
        print(f"Verify token status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Token verification successful")
            verify_data = response.json()
            print(f"Verification result: {verify_data}")
        else:
            print(f"❌ Token verification failed: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Token verification request failed: {e}")

def test_database():
    """Test database connection and user creation"""
    print("\n🔍 Testing database functionality...")
    try:
        from database import SessionLocal, engine
        from models import User
        from sqlalchemy import text
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        
        # Test user creation
        db = SessionLocal()
        try:
            # Check if test user exists
            existing_user = db.query(User).filter(User.username == "testuser").first()
            if existing_user:
                print(f"✅ Test user exists in database: {existing_user.username}")
            else:
                print("⚠️ Test user not found in database")
            
            # List all users
            all_users = db.query(User).all()
            print(f"📊 Total users in database: {len(all_users)}")
            for user in all_users:
                print(f"  - {user.username} ({user.email})")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")

def test_password_hashing():
    """Test password hashing functionality"""
    print("\n🔍 Testing password hashing...")
    try:
        from routers.auth import get_password_hash, verify_password
        
        test_password = "testpassword123"
        hashed = get_password_hash(test_password)
        print(f"✅ Password hashed successfully")
        print(f"Hashed password: {hashed[:50]}...")
        
        # Test verification
        is_valid = verify_password(test_password, hashed)
        if is_valid:
            print("✅ Password verification successful")
        else:
            print("❌ Password verification failed")
            
    except Exception as e:
        print(f"❌ Password hashing test failed: {e}")

def main():
    """Main debugging function"""
    print("🚀 Starting login debugging process...")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    # Test server connection
    if not test_server_connection():
        print("\n❌ Server is not running. Please start the backend server first.")
        print("Run: cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Test database
    test_database()
    
    # Test password hashing
    test_password_hashing()
    
    # Test authentication endpoints
    token = test_auth_endpoints()
    
    # Test protected endpoints
    test_protected_endpoints(token)
    
    print("\n" + "=" * 50)
    print("🏁 Debugging complete!")
    
    if token:
        print("✅ Login functionality is working correctly!")
        print("\nTo test in the frontend:")
        print("1. Make sure the frontend is running on http://localhost:3000")
        print("2. Try logging in with username: 'testuser' and password: 'testpassword123'")
    else:
        print("❌ Login functionality has issues that need to be resolved.")
        print("\nCommon issues:")
        print("1. Check if environment variables are set correctly")
        print("2. Verify database tables are created")
        print("3. Check server logs for errors")

if __name__ == "__main__":
    main()
