# Email Functionality Setup Guide

## 📧 Overview

The AI Web Scraper now includes email functionality that automatically sends welcome emails to new users upon successful registration. This feature works on both localhost and deployed platforms.

## 🚀 Features

- **Welcome Email**: Automatically sent to new users after successful registration
- **Beautiful HTML Templates**: Professional-looking emails with responsive design
- **Fallback Support**: Registration succeeds even if email sending fails
- **Gmail Integration**: Optimized for Gmail SMTP
- **Security**: Uses App Passwords for enhanced security

## ⚙️ Configuration

### 1. Gmail Setup (Recommended)

1. **Enable 2-Factor Authentication**
   - Go to your Google Account settings
   - Navigate to Security → 2-Step Verification
   - Enable 2-factor authentication

2. **Generate App Password**
   - Go to Security → 2-Step Verification → App passwords
   - Select "Mail" and your device
   - Copy the generated 16-character password

3. **Update Environment Variables**
   ```bash
   # In backend/.env file
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SENDER_EMAIL=your_gmail_address@gmail.com
   SENDER_PASSWORD=your_16_character_app_password
   SENDER_NAME=AI Web Scraper
   ```

### 2. Other Email Providers

For other email providers, update the SMTP settings accordingly:

```bash
# Example for Outlook/Hotmail
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SENDER_EMAIL=your_email@outlook.com
SENDER_PASSWORD=your_app_password
SENDER_NAME=AI Web Scraper
```

## 🧪 Testing

### 1. Test Email Configuration

Run the email registration test:

```bash
cd backend
python test_email_registration.py
```

### 2. Manual Testing

1. Start the application:
   ```bash
   # Backend
   cd backend
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   
   # Frontend
   cd frontend
   npm start
   ```

2. Register a new user through the web interface
3. Check your email inbox for the welcome message
4. Check backend terminal for email sending logs

### 3. Expected Logs

Successful email sending:
```
INFO:services.email_service:Email service configured with sender: your_email@gmail.com
Welcome email sent successfully to user@example.com
```

Failed email sending (but registration still succeeds):
```
Failed to send welcome email to user@example.com
Error sending welcome email to user@example.com: [error details]
```

## 🔧 Troubleshooting

### Common Issues

1. **"Authentication failed" Error**
   - Make sure you're using an App Password, not your regular Gmail password
   - Verify 2-factor authentication is enabled
   - Check that the email address is correct

2. **"Connection refused" Error**
   - Verify SMTP server and port settings
   - Check firewall settings
   - Ensure internet connectivity

3. **Email not received**
   - Check spam/junk folder
   - Verify the recipient email address
   - Check email provider's delivery logs

### Debug Mode

To see detailed email logs, check the backend terminal output when registering users. The system will log:
- Email service configuration status
- Email sending attempts
- Success/failure messages
- Detailed error information

## 🚀 Deployment

### Environment Variables for Production

When deploying to production platforms (Heroku, Railway, Vercel, etc.), set these environment variables:

```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_production_email@gmail.com
SENDER_PASSWORD=your_app_password
SENDER_NAME=Your App Name
```

### Security Best Practices

1. **Never commit credentials** to version control
2. **Use App Passwords** instead of regular passwords
3. **Rotate passwords** regularly
4. **Use environment variables** for all sensitive data
5. **Monitor email sending** for abuse

## 📋 Email Template Customization

The welcome email template is located in `backend/services/email_service.py`. You can customize:

- Email subject line
- HTML template design
- Plain text fallback
- Sender name
- Call-to-action links

### Template Variables

Available variables in email templates:
- `{{ username }}`: User's username
- `{{ email }}`: User's email address
- Custom variables can be added as needed

## 🔄 Future Enhancements

Planned email features:
- Password reset emails
- Job completion notifications
- Weekly digest emails
- Email preferences management
- Email analytics and tracking

## 📞 Support

If you encounter issues with email functionality:

1. Check this documentation
2. Review the troubleshooting section
3. Check backend logs for detailed error messages
4. Verify your email provider's SMTP settings
5. Test with a different email address

## 🎯 Quick Start Checklist

- [ ] Enable 2-factor authentication on Gmail
- [ ] Generate Gmail App Password
- [ ] Update `.env` file with email credentials
- [ ] Restart backend server
- [ ] Test registration with a new user
- [ ] Check email inbox for welcome message
- [ ] Verify backend logs show successful email sending

---

**Note**: Email functionality is designed to be non-blocking. User registration will always succeed, even if email sending fails. This ensures a smooth user experience while providing the email enhancement when properly configured.