from pathlib import Path

def test_required_project_files_exist():
    required = [
        "README.md", "DATA_ACCESS.md", "requirements.txt",
        "src/model.py", "src/data.py",
        "notebooks/dental_xray_autoencoder_denoising.ipynb",
    ]
    for path in required:
        assert Path(path).exists(), f"Missing {path}"
