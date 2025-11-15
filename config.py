import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 설정
PROJECT_ROOT = Path(__file__).parent.resolve()

def get_FILE_NAME():
    # Check environment variable first
    if FILE_NAME_env := os.getenv("FILE_NAME"):
        return FILE_NAME_env
    
    # Auto-detect from data directory
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
        if pdf_files:
            # Use the first PDF found (or you could use the most recent)
            # Remove .pdf extension to get the name
            return pdf_files[0].stem
    
    # Error handling: no PDF found and no environment variable set
    raise ValueError(
        "FILE_NAME could not be determined. "
        "Please either:\n"
        "  1. Set the FILE_NAME environment variable: export FILE_NAME='your_FILE_NAME'\n"
        "  2. Place a PDF file in the data/ directory"
    ) 

FILE_NAME = get_FILE_NAME()
input_file = PROJECT_ROOT / "data" / f"{FILE_NAME}.pdf"  # Replace with a file of your own
batch_size = 10  # Maximum available value is 100
image_output_path_prefix = f"/images/{FILE_NAME}_cropped"
output_path_prefix = PROJECT_ROOT / "outputs" / f"{FILE_NAME}_output"