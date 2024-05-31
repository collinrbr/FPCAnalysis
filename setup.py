import os
import sys
from setuptools import setup

def create_virtualenv():
    python_version = sys.version_info
    virtualenv_command = f"python -m venv myenv"
    os.system(virtualenv_command)

def install_required_libraries():
    with open('requirements.txt') as f:
        required_libraries = f.read().splitlines()
        for library in required_libraries:
            os.system(f"myenv/bin/pip install {library}")

def main():
    create_virtualenv()
    install_required_libraries()

if __name__ == "__main__":
    main()

