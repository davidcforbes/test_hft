"""
Create a simple test PDF for testing the PDF parsing functionality.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf(filename="test_document.pdf"):
    """Create a simple test PDF with text content."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Page 1
    c.drawString(100, height - 100, "Test Document - Page 1")
    c.drawString(100, height - 150, "This is a sample PDF document for testing")
    c.drawString(100, height - 200, "the PDF parsing functionality with PyMuPDF.")
    c.drawString(100, height - 250, "This page contains some text content.")
    c.showPage()
    
    # Page 2
    c.drawString(100, height - 100, "Test Document - Page 2")
    c.drawString(100, height - 150, "This is the second page of the test document.")
    c.drawString(100, height - 200, "It also contains text that will be converted")
    c.drawString(100, height - 250, "to images for vision model processing.")
    c.showPage()
    
    c.save()
    print(f"Created test PDF: {filename}")

if __name__ == "__main__":
    create_test_pdf()