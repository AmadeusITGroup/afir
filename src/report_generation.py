import asyncio
import io
import json
import logging
import os.path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

from src.utils.error_handling import retry_with_backoff
from src.utils.llm_utils import get_llm_response

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

        For each anomaly, provide a comprehensive and detailed analysis, making sure to be thorough and step-by-step 
        in your explanations. Each subsection should be explained in full, with a clear, conversational tone. Here’s 
        the format to follow for each anomaly: 
        - Description: Provide a detailed overview of the anomaly. Be clear and specific about what the anomaly is and any relevant context that helps in understanding it.
        - Reason of detection:  Explain in depth why this was detected as an anomaly. Guide the reader step-by-step through the reasoning, making sure to break down any criteria, patterns, or unusual behaviors that flagged this as significant.
        - Supporting Evidence: List all pieces of evidence used to identify the anomaly, providing a brief description for each. Make sure to explain how the pieces of information combine or interact to reinforce the detection, using a step-by-step rationale.
        - Potential implications: Outline the potential impacts of this anomaly if left unaddressed. Discuss any risks, consequences, or changes it could trigger, explaining each in full to provide a clear picture of its significance.
        - Confidence score: Assign a confidence score to this detection. Explain the reasoning behind this score, covering all relevant factors that contribute to your level of certainty.
        - Recommended investigation or mitigation steps: Lay out step-by-step actions that could be taken to investigate or mitigate this anomaly. Provide rationale for each recommended step, explaining how it could help further clarify or resolve the issue.
        Make sure to approach each section in a way that thoroughly explains each piece of reasoning, building a clear and detailed narrative for every part of the analysis.
     
        
        Keep in mind these definitions: 
        - section: a dictionary with only two fields called "section_title" and "content".
        - subsection: a dictionary with only two fields called "section_title" and "content". 
        - content: can only be a string, a list of strings, or a list of subsections (i.e. list of dictionaries).

        Format the output as an array of JSONs. The JSONs will be sections, referring to the above definitions. 
        No other types of data or structure should be part of the JSON.
        
        Include relevant statistics, key metrics, and potential business impact where applicable.
        
        """

    def append_pdf_content(self, story, content, styles):
        if isinstance(content, str) or isinstance(content, int) or isinstance(content, float):
            story.append(Paragraph(str(content), styles['Normal']))
        elif isinstance(content, dict):
            for subsection, subcontent in content.items():
                if subsection in ["section_title", "subsection_title"]:
                    story.append(Paragraph(subcontent, styles['Heading2']))
                elif subsection in ["sections", "section", "subsection", "content"]:
                    self.append_pdf_content(story, subcontent, styles)
                else:
                    story.append(Paragraph(subsection, styles['Heading2']))
                    self.append_pdf_content(story, subcontent, styles)
        elif isinstance(content, list):
            for item in content:
                self.append_pdf_content(story, item, styles)
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
            print(report_content)
            structured_report = json.loads(report_content)
            logger.info(f"Correctly generated report content...")

            if self.config['output_format'] == 'pdf':
                report = self.generate_pdf_report(structured_report, incident, anomalies)
                self.export_report(report, 'pdf', self.config['output_path'])
                return report
            else:
                report = json.dumps(structured_report, indent=2)
                self.export_report(report, 'txt', self.config['output_path'])
                return report
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
            logo = Image(self.config['logo_path'], width=250, height=40)
            story.append(logo)
            story.append(Spacer(1, 12))

        # Add title
        story.append(Paragraph(f"Fraud Investigation Report - Incident {incident['id']}", styles['Title']))
        story.append(Spacer(1, 12))

        # Add content
        for section in structured_report:
            story = self.append_pdf_content(story, section, styles)
            story.append(Spacer(1, 12))

        # Add anomalies table
        story.append(Paragraph("Detailed Anomalies", styles['Heading1']))
        story.append(Spacer(1, 6))

        anomalies_data = [["Description", "Confidence", "Potential Implications", "Recommended Actions"]]
        for anomaly in anomalies:
            anomalies_data.append([
                anomaly['description'],
                f"{anomaly['confidence_score']:.2f}",
                anomaly['potential_implications'],
                anomaly['recommended_actions']
            ])

        page_width, page_height = A4
        left_margin = 0.5 * inch
        right_margin = 0.5 * inch
        available_width = page_width - left_margin - right_margin
        column_count = len(anomalies_data[0])
        column_widths = [available_width / column_count] * column_count
        styles = getSampleStyleSheet()
        style_normal = styles['Normal']

        wrapped_data = [
            [Paragraph(str(cell), style_normal) for cell in row]
            for row in anomalies_data
        ]

        anomalies_table = Table(wrapped_data, colWidths=column_widths)
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
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(anomalies_table)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def export_report(self, report, file_type, path):
        if file_type == 'pdf':
            with open(os.path.join(path, "fraud_report.pdf"), 'wb') as f:
                f.write(report)
            print("Report generated and saved")
        else:
            with open(os.path.join(path, "fraud_report.txt"), 'w') as f:
                f.write(report)


# Example usage
async def main():
    config = {
        'output_path': '../exports',
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
                'max_tokens': 2000,
                'temperature': 0.7
            }
        }
    }

    incident = {
        'id': 'INC-12345',
        'timestamp': '2023',
        'description': 'Unusual login activity detected from multiple IP addresses.'
    }

    understanding = {
        'incident_id': 'INC-12345',
        'analysis': 'Potential unauthorized access detected with multiple failed login attempts followed by a '
                    'successful login from an unrecognized IP address.'
    }

    anomalies = [
        {
            'description': 'Multiple failed login attempts from various IP addresses',
            'confidence_score': 0.95,
            'potential_implications': 'Possible brute force attack attempt',
            'recommended_actions': 'Implement IP-based rate limiting and notify the account owner'
        },
        {
            'description': 'Successful login from an unrecognized IP address after failed attempts',
            'confidence_score': 0.85,
            'potential_implications': 'Potential account compromise',
            'recommended_actions': 'Force password reset and enable two-factor authentication'
        }
    ]

    report_generation = ReportGenerationModule(config, llm_config)
    report = await report_generation.generate(incident, understanding, {}, anomalies)


if __name__ == "__main__":
    asyncio.run(main())
