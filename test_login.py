#!/usr/bin/env python3
"""
Simple login test script
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000/api"

def test_login():
    print("Testing login functionality...")
    
    # Test server connection
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"Server status: {response.status_code}")
    except Exception as e:
        print(f"Server connection failed: {e}")
        return False
    
    # Test registration
    print("\nTesting registration...")
    register_data = {
        "username": "testuser",
        "email": "test@example.com", 
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=register_data, timeout=10)
        print(f"Registration status: {response.status_code}")
        if response.status_code == 200:
            print("Registration successful")
        elif response.status_code == 400:
            print("User might already exist (expected)")
        else:
            print(f"Registration failed: {response.text}")
    except Exception as e:
        print(f"Registration error: {e}")
    
    # Test login
    print("\nTesting login...")
    login_data = {
        "username": "testuser",
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=10)
        print(f"Login status: {response.status_code}")
        if response.status_code == 200:
            print("Login successful!")
            token_data = response.json()
            print(f"Token received: {token_data.get('access_token', 'No token')[:50]}...")
            return True
        else:
            print(f"Login failed: {response.text}")
            return False
    except Exception as e:
        print(f"Login error: {e}")
        return False

if __name__ == "__main__":
    success = test_login()
    if success:
        print("\nLogin functionality is working!")
        print("You can now test in the frontend with:")
        print("Username: testuser")
        print("Password: testpassword123")
    else:
        print("\nLogin functionality has issues.")
