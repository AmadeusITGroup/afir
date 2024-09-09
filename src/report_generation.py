import asyncio
import json
from src.utils.llm_utils import get_llm_response
from src.utils.error_handling import retry_with_backoff
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)


class ReportGenerationModule:
    def __init__(self, config, llm_config):
        self.config = config
        self.llm_config = llm_config
        self.prompt_template = """
        Generate a comprehensive fraud investigation report based on the following information:

        Incident:
        {incident}

        Understanding:
        {understanding}

        Anomalies:
        {anomalies}

        Please provide a detailed report including:
        1. Executive Summary
        2. Incident Overview
        3. Investigation Process
        4. Detailed Findings and Anomalies Analysis
        5. Risk Assessment
        6. Recommended Actions
        7. Conclusion
        8. Next Steps

        For each anomaly, include:
        - Description
        - Supporting evidence
        - Potential implications
        - Confidence score
        - Recommended investigation or mitigation steps

        Format the output as a structured JSON object with sections and subsections.
        Include relevant statistics, key metrics, and potential business impact where applicable.
        Make sure to use escaping with all the symbols that may be a problem if I am using APIs.
        """

    def append_pdf_content(self, story, content, styles):
        if isinstance(content, str):
            story.append(Paragraph(content, styles['Justify']))
        elif isinstance(content, dict):
            for subsection, subcontent in content.items():
                story.append(Paragraph(subsection, styles['Heading2']))
                self.append_pdf_content(story, subcontent, styles)
        return story

    @retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def generate(self, incident, understanding, logs, anomalies):
        try:
            prompt = self.prompt_template.format(
                incident=json.dumps(incident, indent=2),
                understanding=understanding['analysis'],
                anomalies=json.dumps(anomalies, indent=2)
            )

            report_content = await get_llm_response(prompt, self.llm_config)
            structured_report = json.loads(report_content)

            if self.config['output_format'] == 'pdf':
                return self.generate_pdf_report(structured_report, incident, anomalies)
            else:
                return json.dumps(structured_report, indent=2)
        except Exception as e:
            logger.error(f"Error generating report for incident {incident['id']}: {str(e)}")
            raise

    def generate_pdf_report(self, structured_report, incident, anomalies):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=1))

        story = []

        # Add logo
        if 'logo_path' in self.config:
            logo = Image(self.config['logo_path'], width=100, height=50)
            story.append(logo)
            story.append(Spacer(1, 12))

        # Add title
        story.append(Paragraph(f"Fraud Investigation Report - Incident {incident['id']}", styles['Title']))
        story.append(Spacer(1, 12))

        # Add content
        for section, content in structured_report.items():
            story.append(Paragraph(section, styles['Heading1']))
            story.append(Spacer(1, 6))

            story = self.append_pdf_content(story, content, styles)

            story.append(Spacer(1, 12))

        # Add anomalies table
        story.append(Paragraph("Detailed Anomalies", styles['Heading1']))
        story.append(Spacer(1, 6))

        anomalies_data = [["Description", "Confidence", "Implications", "Recommended Actions"]]
        for anomaly in anomalies:
            anomalies_data.append([
                anomaly['description'],
                f"{anomaly['confidence_score']:.2f}",
                anomaly['implications'],
                anomaly['recommended_actions']
            ])

        anomalies_table = Table(anomalies_data)
        anomalies_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(anomalies_table)

        # Add visualizations
        if self.config['include_visualizations']:
            story.append(Paragraph("Visualizations", styles['Heading1']))
            story.append(Spacer(1, 6))

            # Anomaly confidence distribution
            confidence_chart = self.create_confidence_chart(anomalies)
            story.append(confidence_chart)
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def create_confidence_chart(self, anomalies):
        confidences = [anomaly['confidence_score'] for anomaly in anomalies]
        plt.figure(figsize=(6, 4))
        plt.hist(confidences, bins=10, edgecolor='black')
        plt.title('Distribution of Anomaly Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        return Image(img_buffer, width=300, height=200)


# Example usage
async def main():
    config = {
        'output_format': 'pdf',
        'logo_path': '../assets/company_logo.jfif',
        'include_visualizations': True
    }

    llm_config = {
        'provider': 'generic',
        "use_fine_tuned": False,
        'models': {
            'default': {
                'name': "gpt-3.5-turbo-0613",
                'api_key': "kQqfUAK1aXZ2SmRgk64is0",
                'max_tokens': 2000,
                'temperature': 0.7
            }
        }
    }

    incident = {
        'id': 'INC-12345',
        'timestamp': '2023-07-06T12:00:00Z',
        'description': 'Unusual login activity detected from multiple IP addresses.'
    }

    understanding = {
        'incident_id': 'INC-12345',
        'analysis': 'Potential unauthorized access detected with multiple failed login attempts followed by a successful login from an unrecognized IP address.'
    }

    anomalies = [
        {
            'description': 'Multiple failed login attempts from various IP addresses',
            'confidence_score': 0.95,
            'implications': 'Possible brute force attack attempt',
            'recommended_actions': 'Implement IP-based rate limiting and notify the account owner'
        },
        {
            'description': 'Successful login from an unrecognized IP address after failed attempts',
            'confidence_score': 0.85,
            'implications': 'Potential account compromise',
            'recommended_actions': 'Force password reset and enable two-factor authentication'
        }
    ]

    report_generation = ReportGenerationModule(config, llm_config)
    report = await report_generation.generate(incident, understanding, {}, anomalies)

    if config['output_format'] == 'pdf':
        with open('fraud_report.pdf', 'wb') as f:
            f.write(report)
        print("Report generated and saved")
    else:
        with open('fraud_report.txt', 'w') as f:
            f.write(report)
        print("Report generated and saved")


if __name__ == "__main__":
    asyncio.run(main())
