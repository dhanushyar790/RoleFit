import io
import PyPDF2
import docx


def extract_text_from_bytes(filename: str, file_bytes: bytes) -> str:
    """Extract text from PDF, DOCX, or TXT bytes based on filename extension."""
    filename = filename.lower()

    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "".join([page.extract_text() or "" for page in reader.pages])

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8")

    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")