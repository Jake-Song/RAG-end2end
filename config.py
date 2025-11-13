from pathlib import Path

# 설정
PROJECT_ROOT = Path(__file__).parent.resolve()

input_file = PROJECT_ROOT / "data" / "SPRI_AI_Brief.pdf"  # Replace with a file of your own
batch_size = 10  # Maximum available value is 100
image_output_path_prefix = PROJECT_ROOT / "images" / "SPRI_AI_Brief_cropped"
output_path_prefix = PROJECT_ROOT / "SPRI_AI_Brief_output"