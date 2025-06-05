import os
import PyPDF2
from io import BytesIO
from typing import Optional

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string, or None if extraction fails
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text.append(page_text)
            return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

async def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """
    Extracts text from PDF bytes (useful for FastAPI UploadFile)
    
    Args:
        pdf_bytes: PDF file contents as bytes
        
    Returns:
        Extracted text as a single string, or None if extraction fails
    """
    try:
        with BytesIO(pdf_bytes) as bytes_io:
            reader = PyPDF2.PdfReader(bytes_io)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from PDF bytes: {e}")
        return None