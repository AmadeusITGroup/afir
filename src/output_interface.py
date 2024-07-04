import asyncio
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import logging
from src.utils.error_handling import retry_with_backoff
import aiofiles
import os

logger = logging.getLogger(__name__)


class OutputInterface:
    def __init__(self, config):
        self.config = config

    @retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def send(self, report, incident_id):
        output_type = self.config['type']
        if output_type == 'email':
            await self.send_email(report, incident_id)
        elif output_type == 'file':
            await self.save_to_file(report, incident_id)
        elif output_type == 'api':
            await self.send_to_api(report, incident_id)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

    async def send_email(self, report, incident_id):
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"Fraud Investigation Report - Incident {incident_id}"
            msg['From'] = self.config['sender_email']
            msg['To'] = ", ".join(self.config['recipients'])
            if 'cc' in self.config:
                msg['Cc'] = ", ".join(self.config['cc'])

            msg.attach(MIMEText(f"Please find attached the fraud investigation report for incident {incident_id}."))

            report_attachment = MIMEApplication(report, _subtype="pdf")
            report_attachment.add_header('Content-Disposition', 'attachment',
                                         filename=f"fraud_report_{incident_id}.pdf")
            msg.attach(report_attachment)

            async with aiosmtplib.SMTP(hostname=self.config['smtp_server'], port=self.config['smtp_port']) as smtp:
                await smtp.starttls()
                await smtp.login(self.config['smtp_username'], self.config['smtp_password'])
                await smtp.send_message(msg)

            logger.info(f"Sent fraud investigation report for incident {incident_id} via email")
        except Exception as e:
            logger.error(f"Error sending email for incident {incident_id}: {str(e)}")
            raise

    async def save_to_file(self, report, incident_id):
        try:
            output_dir = self.config.get('output_directory', 'reports')
            os.makedirs(output_dir, exist_ok=True)

            filename = f"fraud_report_{incident_id}.pdf"
            filepath = os.path.join(output_dir, filename)

            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(report)

            logger.info(f"Saved fraud investigation report for incident {incident_id} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving report to file for incident {incident_id}: {str(e)}")
            raise

    async def send_to_api(self, report, incident_id):
        try:
            import aiohttp

            url = self.config['api_url']
            headers = self.config.get('api_headers', {})

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data={'incident_id': incident_id}, files={'report': report},
                                        headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent fraud investigation report for incident {incident_id} to API")
                    else:
                        raise Exception(f"API returned status code {response.status}")
        except Exception as e:
            logger.error(f"Error sending report to API for incident {incident_id}: {str(e)}")
            raise


# Example usage
async def main():
    config = {
        'type': 'email',
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'smtp_username': 'your_email@gmail.com',
        'smtp_password': 'your_password',
        'sender_email': 'fraud_detection@example.com',
        'recipients': ['analyst1@example.com', 'analyst2@example.com'],
        'cc': ['manager@example.com'],
        'output_directory': 'fraud_reports',
        'api_url': 'https://api.example.com/submit_report',
        'api_headers': {'Authorization': 'Bearer your_api_token'}
    }

    output_interface = OutputInterface(config)

    # Simulating a report (in a real scenario, this would be the PDF content)
    report = b"This is a sample PDF report content"
    incident_id = "12345"

    # Test email sending
    await output_interface.send(report, incident_id)

    # Change output type to file saving
    config['type'] = 'file'
    output_interface = OutputInterface(config)
    await output_interface.send(report, incident_id)

    # Change output type to API sending
    config['type'] = 'api'
    output_interface = OutputInterface(config)
    await output_interface.send(report, incident_id)


if __name__ == "__main__":
    asyncio.run(main())
