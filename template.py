import os

folders = [
    "data",
    "data/images",
    "saved_models",
    "src"
]

files = {
    "data/train.csv": "",
    "data/test.csv": "",
    "src/__init__.py": "",
    "src/dataset.py": "# dataset.py — handles data loading, transformations, and augmentation.\n",
    "src/model.py": "# model.py — defines your model architecture and training utilities.\n",
    "src/utils.py": "# utils.py — helper functions for metrics, logging, etc.\n",
    "main_train.py": "# main_train.py — script to train your model.\n",
    "main_inference.py": "# main_inference.py — script to run inference and generate submission.\n",
    "requirements.txt": "# requirements.txt — add dependencies here.\n"
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f" Created folder: {folder}")

    for file, content in files.items():
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f" Created file: {file}")

if __name__ == "__main__":
    create_structure()
    print("\n Project structure successfully created!")
