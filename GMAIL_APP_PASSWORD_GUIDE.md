# 📧 Gmail App Password Setup Guide

## 🚨 IMPORTANT: You need a REAL Gmail App Password for emails to work!

The current configuration uses a placeholder password (`1234567890123456`). Follow these steps to get your real Gmail App Password:

## 📋 Step-by-Step Instructions

### 1. Enable 2-Factor Authentication
1. Go to [Google Account Settings](https://myaccount.google.com/)
2. Click on **Security** in the left sidebar
3. Under "Signing in to Google", click **2-Step Verification**
4. If not enabled, click **Get Started** and follow the setup process
5. Verify your phone number and complete the setup

### 2. Generate App Password
1. After 2-Factor Authentication is enabled, go back to **Security**
2. Under "Signing in to Google", click **2-Step Verification**
3. Scroll down and click **App passwords**
4. You might need to sign in again
5. In the "Select app" dropdown, choose **Mail**
6. In the "Select device" dropdown, choose **Other (Custom name)**
7. Type: `AI Web Scraper` or any name you prefer
8. Click **Generate**
9. **COPY THE 16-CHARACTER PASSWORD** (it will look like: `abcd efgh ijkl mnop`)

### 3. Update Your Configuration
Run this command in the backend directory:
```bash
python setup_email.py
```

When prompted:
- **Gmail address**: `svaidik54@gmail.com` (already configured)
- **App Password**: Paste the 16-character password you just generated
- **Sender name**: `AI Web Scraper` (or your preferred name)

### 4. Test Email Functionality
After updating with the real App Password:

1. **Restart the backend server** (it should restart automatically)
2. **Register a new user** through the web interface
3. **Check your mobile email** for the welcome message
4. **Check backend logs** for confirmation

## 🔍 Troubleshooting

### If you can't find "App passwords":
- Make sure 2-Factor Authentication is fully enabled
- Wait a few minutes after enabling 2FA
- Try signing out and back into your Google Account

### If emails still don't work:
- Verify the App Password is exactly 16 characters
- Make sure you're using `svaidik54@gmail.com` (your Gmail address)
- Check if Gmail is blocking the app (check your Gmail security notifications)

### Common App Password formats:
- Correct: `abcd efgh ijkl mnop` (16 characters with or without spaces)
- Incorrect: Your regular Gmail password
- Incorrect: Random numbers like `1234567890123456`

## 🧪 Quick Test

After setting up the real App Password, run this test:

```bash
cd backend
python quick_test.py
```

This will:
1. Register a test user
2. Trigger the welcome email
3. Show you the backend logs
4. Confirm if the email was sent successfully

## 📱 Expected Result

Once configured correctly, you should see in the backend logs:
```
INFO:services.email_service:Email service configured with sender: svaidik54@gmail.com
INFO:services.email_service:Welcome email sent successfully to test@example.com
```

And you should receive a beautiful welcome email on your mobile device! 🎉

## 🔐 Security Notes

- **Never share your App Password** with anyone
- **Don't commit it to version control** (it's in .env which is gitignored)
- **You can revoke and regenerate** App Passwords anytime from Google Account settings
- **Each app should have its own** App Password for security

---

**Need help?** Check the backend terminal logs for detailed error messages after testing!