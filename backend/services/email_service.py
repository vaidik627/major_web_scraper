import os
from typing import Optional
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.app_name = os.getenv("APP_NAME", "AI Web Scraper")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.sender_email = os.getenv("SENDER_EMAIL", "noreply@aiwebscraper.com")
        self.sender_name = os.getenv("SENDER_NAME", "AI Web Scraper")
        
        # Initialize SendGrid client if API key is available and valid
        self.sg_client = None
        if (self.sendgrid_api_key and 
            self.sendgrid_api_key != "your_sendgrid_api_key_here" and 
            self.sendgrid_api_key.startswith("SG.")):
            try:
                self.sg_client = SendGridAPIClient(api_key=self.sendgrid_api_key)
                logger.info("✅ SendGrid client initialized successfully - Real emails enabled")
            except Exception as e:
                logger.error(f"❌ Failed to initialize SendGrid client: {str(e)}")
                logger.warning("📧 Falling back to console email mode")
        else:
            logger.warning("📧 SENDGRID_API_KEY not configured. Using console email mode.")
        
    def _send_email(self, to_email: str, subject: str, html_content: str) -> bool:
        """
        Internal method to send email via SendGrid or fallback to console
        """
        try:
            if self.sg_client:
                # Send real email via SendGrid
                message = Mail(
                    from_email=(self.sender_email, self.sender_name),
                    to_emails=to_email,
                    subject=subject,
                    html_content=html_content
                )
                
                response = self.sg_client.send(message)
                
                if response.status_code in [200, 201, 202]:
                    logger.info(f"✅ Real email sent successfully to {to_email} (Status: {response.status_code})")
                    return True
                else:
                    logger.error(f"❌ SendGrid API error: {response.status_code}")
                    logger.error(f"Response body: {response.body}")
                    logger.error(f"Response headers: {response.headers}")
                    return False
            else:
                # Fallback to console logging
                print("\n" + "="*60)
                print("📧 EMAIL NOTIFICATION (Console Mode)")
                print("="*60)
                print(f"📬 To: {to_email}")
                print(f"📝 Subject: {subject}")
                print("📄 Content:")
                print("-" * 40)
                # Extract text content from HTML for better console display
                import re
                text_content = re.sub('<[^<]+?>', '', html_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
                print("-" * 40)
                print("💡 To enable real emails: Configure SENDGRID_API_KEY in .env")
                print("="*60)
                logger.info(f"📧 Console email logged for {to_email}")
                return True
                
        except Exception as e:
            # Check if it's a SendGrid specific error
            if hasattr(e, 'status_code'):
                logger.error(f"SendGrid HTTP Error {e.status_code}: {str(e)}")
                if hasattr(e, 'body'):
                    logger.error(f"Error body: {e.body}")
            else:
                logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
        
    def send_welcome_email(self, email: str, username: str) -> bool:
        """
        Send a welcome email to new users
        """
        try:
            subject = f"Welcome to {self.app_name}! 🎉"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Welcome to {self.app_name}</title>
            </head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                    <h1 style="color: white; margin: 0; font-size: 28px;">🎉 Welcome to {self.app_name}!</h1>
                </div>
                
                <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; border: 1px solid #e9ecef;">
                    <h2 style="color: #495057; margin-top: 0;">Hi {username}! 👋</h2>
                    
                    <p style="font-size: 16px; margin-bottom: 20px;">
                        Your account has been successfully created! Thank you for joining {self.app_name}.
                    </p>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea;">
                        <h3 style="margin-top: 0; color: #495057;">🚀 What you can do now:</h3>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li>📊 Track trends and analytics</li>
                            <li>💾 Save and export your data</li>
                            <li>🤖 Use AI-powered insights</li>
                            <li>📈 Monitor web scraping jobs</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{self.frontend_url}/login" 
                           style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                  color: white; 
                                  padding: 12px 30px; 
                                  text-decoration: none; 
                                  border-radius: 25px; 
                                  font-weight: bold; 
                                  display: inline-block;">
                            🚀 Get Started Now
                        </a>
                    </div>
                    
                    <p style="font-size: 14px; color: #6c757d; margin-top: 30px; text-align: center;">
                        If you have any questions, feel free to reach out to our support team.<br>
                        <strong>{self.app_name} Team</strong>
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(email, subject, html_content)
            
        except Exception as e:
            logger.error(f"Failed to send welcome email to {email}: {str(e)}")
            return False
    

    
    def send_login_notification(self, email: str, username: str) -> bool:
        """
        Send a login notification email
        """
        try:
            subject = f"🔐 Security Alert: New Login to {self.app_name}"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Security Alert</title>
            </head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                    <h1 style="color: white; margin: 0; font-size: 24px;">🔐 Security Alert</h1>
                </div>
                
                <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; border: 1px solid #e9ecef;">
                    <h2 style="color: #495057; margin-top: 0;">Hi {username}! 👋</h2>
                    
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107;">
                        <p style="margin: 0; font-weight: bold; color: #856404;">
                            🚨 We detected a new login to your {self.app_name} account.
                        </p>
                    </div>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h3 style="margin-top: 0; color: #495057;">Login Details:</h3>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li>⏰ <strong>Time:</strong> Just now</li>
                            <li>👤 <strong>Account:</strong> {email}</li>
                            <li>🌐 <strong>Service:</strong> {self.app_name}</li>
                        </ul>
                    </div>
                    
                    <div style="background: #d1ecf1; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #17a2b8;">
                        <p style="margin: 0; color: #0c5460;">
                            ✅ <strong>If this was you:</strong> No action needed.<br>
                            ❌ <strong>If this wasn't you:</strong> Please secure your account immediately.
                        </p>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{self.frontend_url}/login" 
                           style="background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); 
                                  color: white; 
                                  padding: 12px 30px; 
                                  text-decoration: none; 
                                  border-radius: 25px; 
                                  font-weight: bold; 
                                  display: inline-block;">
                            🔒 Secure My Account
                        </a>
                    </div>
                    
                    <p style="font-size: 14px; color: #6c757d; margin-top: 30px; text-align: center;">
                        Stay secure,<br>
                        <strong>{self.app_name} Security Team</strong>
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(email, subject, html_content)
            
        except Exception as e:
            logger.error(f"Failed to send login notification to {email}: {str(e)}")
            return False

# Create global email service instance
email_service = EmailService()