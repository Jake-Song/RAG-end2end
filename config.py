from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 설정
PROJECT_ROOT = Path(__file__).parent.resolve()

FILE_NAME = "SPRI_2025"
input_file = PROJECT_ROOT / "data" / f"{FILE_NAME}.pdf"  # Replace with a file of your own
batch_size = 10  # Maximum available value is 100
image_output_path_prefix = f"/images/{FILE_NAME}_cropped"
output_path_prefix = PROJECT_ROOT / "outputs" / f"{FILE_NAME}_output"