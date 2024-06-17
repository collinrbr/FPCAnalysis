#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil

def check_python_version(version):
    python_executable = f"python{version}"  # Modify the prefix as needed
    if os.system(f"{python_executable} --version > /dev/null 2>&1") != 0:
        print(f"Python {version} is not installed.")
        print("Please install python3.11! (Try 'sudo apt install python3.11' for linux and 'brew install python@3.11' for mac (requires brew!)'")
        print("Note, you may need to update first with 'sudo apt update' then 'sudo apt install software-properties-common' followed by sudo add-apt-repository ppa:deadsnakes/ppa' and 'sudo apt update' to be able to install python 3.11 using command above!")
        print("To test if install worked, try 'python3.11'")
        exit()
    else:
        print(f"Python {version} is installed.")

def install_required_libraries(env_name):

    #Create conda environment with some dependencies
    print("Creating new conda environment with dependencies")
    command = ["conda", "env", "create", "-f", "environment.yml", "--prefix", "FPCAnalysisenv", "--verbose"]
    try:
        subprocess.run(command, check=True)
        print("Environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Exiting...")
        exit()

    try:
        # Activate the virtual environment and install requirements
        print("Activating the new conda env and installing requirements....")
        requirements_file = 'requirements.txt'
        if os.name == 'nt':  # Windows
            pip_executable = os.path.join(env_name, 'Scripts', 'pip')
        else:  # Unix/Linux/MacOS
            pip_executable = os.path.join(env_name, 'bin', 'pip')

        subprocess.run([pip_executable, 'install', '--no-cache-dir', '-r', requirements_file])

        print("Packages installed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Exiting...")
        exit()

    # Install postgkeyll
    print("Installing postgkeyll (Warning: we install a specific version- not the most current version!)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    subprocess.run(['git', 'clone', 'https://github.com/ammarhakim/postgkyl.git'])
    os.chdir('postgkyl')
    subprocess.run(['git', 'checkout', 'f876908d9e0969e7608100c7b769410e21c56549'])
    os.chdir(os.path.join(script_dir, '..'))
    if os.path.exists('FPCAnalysis/postgkyl'):
        confirmation = input("The directory 'FPCAnalysis/postgkyl' already exists. Do you want to overwrite it? (y/n): ")
        if confirmation.lower() == 'y':
            shutil.rmtree('FPCAnalysis/postgkyl', ignore_errors=True)
            os.rename('postgkyl', 'FPCAnalysis/postgkyl')
            print("Directory overwritten successfully.")
        else:
            print("Not overwriten! Will compile the postgkyl that is already there instead...")
    else:
        os.rename('postgkyl', 'FPCAnalysis/postgkyl')
    os.chdir(os.path.join(script_dir, '..', 'FPCAnalysis'))
    subprocess.run([pip_executable, 'install', '-e', 'postgkyl', '--verbose'])
    print("Done installing postgkeyll!")

    #Install FPCAnalysis lib
    print("Installing FPCAnalysis library into env!")
    try:
        # Run the pip install -e . command with verbose output
        result = subprocess.run([pip_executable, 'install', '-e', '.', '--verbose'], check=True, text=True)
        print(result)
        print("Successfully installed in editable mode.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing the package.")
        print(e)

    print("Done installing FPCAnalysis into env!")

def check_latex_installed():
    # Check if pdflatex is in the PATH
    pdflatex_path = shutil.which("pdflatex")
    if pdflatex_path:
        print(f"pdflatex is installed at {pdflatex_path}")
        return True
    else:
        print("pdflatex is not installed.")
        return False

def main():
    env_name = 'FPCAnalysisenv'
    python_version = '3.11'  # Specify the Python version you want here
    check_python_version(python_version)
    install_required_libraries(env_name)

if __name__ == "__main__":
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help!")

    import time
    time.sleep(5)

    main()

    if not check_latex_installed():
        print();print();print();print();
        print();print();print();print();
        print("LaTeX is required for this script to work with matplotlib.")
        print("Please install LaTeX. For example, on Ubuntu you can run:")
        print("  sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended")
        print();print();print();print();
        print();print();print();print();

        time.sleep(10)

    print();print();print();print();
    print("Completed! Installation")
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help")
    print("Please activate the new enviroment! 'conda activate /path/to/here/FPCAnalysisenv' for linux/mac")
    print("This will need to be done every time a new terminal is launched!!!")

    print('*');print('*');print('*');print('*');
    print("Done with install... Please run `conda actiavte /path/to/here/FPCAnalysisenv' to activate the library or add '#!/path/to/here/FPCAnalysisenv/bin/python' (if on linux/mac) to the top of all scripts (and run by calling ./*scriptname.py)!")
    print("When done, use conda deactivate to turn off environemnt,")
    print("If using the environemnt, you will need to reactivate it every time you open a new terminal if you want to use the FPCAnalysis lib.")
    print("Be sure to activate the environemnt before launching a new jupyter notebook!")
    print('*');print('*');print('*');print('*');

    #General fixes!
    # Always a good idea to update pip 'python -m pip install --upgrade pip'
    #     and to update conda with 'conda update conda'

    #Ubuntu fixes!
    #If you get "FileNotFoundError: [Errno 2] No such file or directory: 'python3.11'" try running:
    #sudo apt-get install python3.11
    
    #If you get "ModuleNotFoundError: No module named 'distutils.util'" try running: 
    #sudo apt-get install --reinstall python3.11-distutils

    #Be sure to have latex installed
    # 'sudo apt install texlive texlive-latex-extra dvipng' or equivalent for OS
