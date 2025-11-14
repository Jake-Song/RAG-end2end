from pathlib import Path

# 설정
PROJECT_ROOT = Path(__file__).parent.resolve()

pdf_name = "sample"
input_file = PROJECT_ROOT / "data" / f"{pdf_name}.pdf"  # Replace with a file of your own
batch_size = 10  # Maximum available value is 100
image_output_path_prefix = f"/images/{pdf_name}_cropped"
output_path_prefix = PROJECT_ROOT / "outputs" / f"{pdf_name}_output"