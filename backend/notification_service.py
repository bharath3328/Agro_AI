import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from backend.config import settings

class NotificationService:
    @staticmethod
    def send_email(to_email: str, subject: str, content: str):
        if not settings.EMAIL_SENDER or not settings.EMAIL_PASSWORD:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = settings.EMAIL_SENDER
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(content, 'plain'))

            server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
            server.starttls()
            server.login(settings.EMAIL_SENDER, settings.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception:
            return False

    @staticmethod
    def send_verification_code(contact_info: str, code: str, via: str = "EMAIL"):
        if via == "EMAIL":
            subject = "AgroAI Verification Code"
            content = f"Your verification code is: {code}. It expires in 10 minutes."
            return NotificationService.send_email(contact_info, subject, content)
        return False
