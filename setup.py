import os
import subprocess
import sys

def create_virtualenv(env_name, python_version):
    # Ensure virtualenv is installed
    subprocess.run([sys.executable, "-m", "pip", "install", "virtualenv"])

    # Create the virtual environment with the specified Python version
    subprocess.run([sys.executable, "-m", "virtualenv", f"--python=python{python_version}", env_name])

def install_required_libraries(env_name):
    # Activate the virtual environment and install the required libraries
    requirements_file = 'requirements.txt'
    if os.name == 'nt':  # Windows
        pip_executable = os.path.join(env_name, 'Scripts', 'pip')
    else:  # Unix/Linux/MacOS
        pip_executable = os.path.join(env_name, 'bin', 'pip')

    subprocess.run([pip_executable, 'install', '-r', requirements_file])

def main():
    env_name = 'FPCAnalysis'
    python_version = '3.8'  # Specify the Python version you want here
    create_virtualenv(env_name, python_version)
    install_required_libraries(env_name)

if __name__ == "__main__":
    main()

