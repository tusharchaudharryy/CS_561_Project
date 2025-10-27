import os

# Define the main project directory
project_name = "smart-pricing-project"

# Define the folder structure
folders = [
    "data/images",
    "features",
    "models",
    "notebooks",
    "src",
    "scripts"
]

# Create the main project directory if it doesn't exist
if not os.path.exists(project_name):
    os.makedirs(project_name)
    print(f"Created project directory: {project_name}")
else:
    print(f"Project directory '{project_name}' already exists.")

# Create the subdirectories
for folder in folders:
    path = os.path.join(project_name, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created/verified directory: {path}")

# Create empty __init__.py in src
init_path = os.path.join(project_name, "src", "__init__.py")
if not os.path.exists(init_path):
    with open(init_path, "w") as f:
        pass # Create an empty file
    print(f"Created file: {init_path}")

# Create placeholder README.md
readme_path = os.path.join(project_name, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write(f"# {project_name}\n\nProject description goes here.")
    print(f"Created file: {readme_path}")

# Create requirements.txt
req_path = os.path.join(project_name, "requirements.txt")
if not os.path.exists(req_path):
    with open(req_path, "w") as f:
        # Add essential libraries, can be expanded later
        f.write("pandas\n")
        f.write("scikit-learn\n")
        f.write("torch\n")
        f.write("torchvision\n")
        f.write("transformers\n")
        f.write("timm\n")
        f.write("lightgbm\n")
        f.write("Pillow\n")
        f.write("tqdm\n")
        f.write("optuna\n") # If using optuna later
    print(f"Created file: {req_path}")

print("\nFolder structure setup complete!")
print(f"Navigate to '{project_name}' to start working.")