"""
Create another test PDF with a different name to verify the naming convention.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_sample_report(filename="sample_report.pdf"):
    """Create a sample report PDF."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Page 1
    c.drawString(100, height - 100, "Sample Report - Executive Summary")
    c.drawString(100, height - 150, "This document contains important business metrics")
    c.drawString(100, height - 200, "and quarterly performance data for analysis.")
    c.showPage()
    
    # Page 2
    c.drawString(100, height - 100, "Sample Report - Financial Data")
    c.drawString(100, height - 150, "Revenue: $1,000,000")
    c.drawString(100, height - 200, "Expenses: $750,000")
    c.drawString(100, height - 250, "Net Profit: $250,000")
    c.showPage()
    
    # Page 3
    c.drawString(100, height - 100, "Sample Report - Conclusion")
    c.drawString(100, height - 150, "The company shows strong performance")
    c.drawString(100, height - 200, "with positive growth trends.")
    c.showPage()
    
    c.save()
    print(f"Created sample report: {filename}")

if __name__ == "__main__":
    create_sample_report()