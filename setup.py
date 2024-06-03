#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil

def check_python_version(version):
    python_executable = f"python{version}"  # Modify the prefix as needed
    if os.system(f"{python_executable} --version > /dev/null 2>&1") != 0:
        print(f"Python {version} is not installed.")
        print("Please install python3.8! (Try 'sudo apt-get install python3.8' for linux and 'brew install python@3.8' for mac (requires brew!)'")
        exit()
    else:
        print(f"Python {version} is installed.")
def create_virtualenv(env_name, python_version):
    # Ensure virtualenv is installed
    subprocess.run([sys.executable, "-m", "pip", "install", "virtualenv"])

    # Create the virtual environment with the specified Python version
    subprocess.run([sys.executable, "-m", "virtualenv", "--python=python"+str(python_version), env_name])

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
    check_python_version(python_version)
    create_virtualenv(env_name, python_version)
    install_required_libraries(env_name)

if __name__ == "__main__":
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help!")

    import time
    time.sleep(5)

    main()
    print("Completed!")
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help")
    print("Please activate the new enviroment! 'source FPCAnalysis/bin/activate' for linux/mac and 'FPCAnalysis\\Scripts\\activate' for windows")
    print("This will need to be done every time a new terminal is launched!!!")

    #Ubuntu fixes!
    #If you get "FileNotFoundError: [Errno 2] No such file or directory: 'python3.8'" try running:
    #sudo apt-get install python3.8
    
    #If you get "ModuleNotFoundError: No module named 'distutils.util'" try running: 
    #sudo apt-get install --reinstall python3.8-distutils
