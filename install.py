#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil

import time

def check_python_version(version_prefix):
    try:
        result = subprocess.run([f"python{version_prefix.split('.')[0]}", "--version"], capture_output=True, text=True)
        installed_version = result.stdout.strip().split()[1]
        if installed_version.startswith(version_prefix):
            print(f"Python {installed_version} is installed.")
        else:
            raise Exception
    except Exception:
        print(f"Python {version_prefix}.* is not installed.")
        print("It is recommended that you install python3.11! Without it, the install may fail.")
        print("If you have 3.11.1 to <3.12, please disregard this message.")
        print("(Try 'sudo apt install python3.11' for linux and 'brew install python@3.11' for mac (requires brew!)')")
        print("Note, you may need to update first with 'sudo apt update' then 'sudo apt install software-properties-common' followed by 'sudo add-apt-repository ppa:deadsnakes/ppa' and 'sudo apt update' to be able to install python 3.11 using the command above!")
        print("To test if the install worked, try 'python3.11'")
        print("Trying to continue anyways....")
        time.sleep(2)

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

    #create string that holds the start of the commands for installing things into environment
    if os.name == 'nt':  # Windows
        pip_executable = os.path.join(env_name, 'Scripts', 'pip')
    else:  # Unix/Linux/MacOS
        pip_executable = os.path.join(env_name, 'bin', 'pip')

    #Make env
    try:
        print("Creating the new conda env and installing requirements into env....")
        requirements_file = 'requirements.txt'
        subprocess.run([pip_executable, 'install', '--no-cache-dir', '-r', requirements_file])
        print("Packages installed successfully.")

    except subprocess.CalledProcessError as e:
        print();print()
        print('----------------------------')
        print(f"Error: {e}")
        print("Exiting...")
        exit()

        print("Please remove FPCAnalysisenv, fix the error, and try again!!!")
        exit()

    # Install postgkeyll
    print("Installing postgkeyll into env... (Warning: we install a specific version- not the most current version!)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    subprocess.run(['git', 'clone', 'https://github.com/ammarhakim/postgkyl.git'])
    os.chdir('postgkyl')
    subprocess.run(['git', 'checkout', 'f876908d9e0969e7608100c7b769410e21c56549']) #Comment out this line to get most current version or update third element in arr to desired version
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
    try:
        subprocess.run([pip_executable, 'install', '-e', 'postgkyl', '--verbose'])
    except subprocess.CalledProcessError as e:
        print();print()
        print('----------------------------')
        print(f"Error: {e}")
        print("Exiting...")
        exit()

        print("Please remove FPCAnalysisenv, fix the error, and try again!!!")
        exit()

    #Install FPCAnalysis lib
    print("Installing FPCAnalysis library into env!")
    try:
        # Run the pip install -e . command with verbose output
        result = subprocess.run([pip_executable, 'install', '-e', '.', '--verbose'], check=True, text=True)
        print(result)
        print("Successfully installed in editable mode.")
    except subprocess.CalledProcessError as e:
        print();print()
        print('----------------------------')
        print("An error occurred while installing the package.")
        print(e)

        print("Please remove FPCAnalysisenv, fix the error, and try again!!!")
        exit()

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
    
import os
import subprocess
import sys

def get_conda_env_path(env_name):
    print(f"Finding the path for the conda environment: {env_name}")
    try:
        result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error listing conda environments: {result.stderr}", file=sys.stderr)
            sys.exit(1)

        for line in result.stdout.splitlines():
            if env_name in line:
                # Assuming the line format is "env_name    /path/to/env"
                print(f"Found environment {env_name}.")
                return line.split()[-1]
        print(f"Environment {env_name} not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error finding conda environment: {e}", file=sys.stderr)
        sys.exit(1)

def get_current_prompt():
    try:
        # Use subprocess to fetch the current PS1 prompt string
        result = subprocess.run(['bash', '-c', 'echo $PS1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error fetching current prompt: {result.stderr}", file=sys.stderr)
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"Error fetching current prompt: {e}", file=sys.stderr)
        return None

def create_custom_prompt_scripts(env_path, env_name):
    activate_dir = os.path.join(env_path, 'etc', 'conda', 'activate.d')
    deactivate_dir = os.path.join(env_path, 'etc', 'conda', 'deactivate.d')
    
    print(f"Creating activate.d directory at: {activate_dir}")
    os.makedirs(activate_dir, exist_ok=True)
    print(f"Creating deactivate.d directory at: {deactivate_dir}")
    os.makedirs(deactivate_dir, exist_ok=True)
    
    custom_prompt_activate_script = os.path.join(activate_dir, 'custom_prompt.sh')
    custom_prompt_deactivate_script = os.path.join(deactivate_dir, 'reset_prompt.sh')

    #TODO: at time that this script is called- the prompt string has been modified and thus OLD_PS1 is not the original
    # Hence why we just set the the deactivate script to a hard coded prompt string.
    # We should save the original prompt string earlier on if possible (perhaps during install) and restore it in deactivate script
    activate_script_content = f"""#!/bin/sh
export OLD_PS1="$PS1"
export PS1='({env_name}) \\u@\\h:\\w\\$ '
"""

    deactivate_script_content = f"""#!/bin/sh
    export PS1='\\u@\\h:\\w\\$ '
"""

    try:
        print(f"Writing custom prompt activate script to: {custom_prompt_activate_script}")
        with open(custom_prompt_activate_script, 'w') as file:
            file.write(activate_script_content)
        print(f"Making the custom prompt activate script executable.")
        os.chmod(custom_prompt_activate_script, 0o755)  # Make the script executable

        print(f"Writing custom prompt deactivate script to: {custom_prompt_deactivate_script}")
        with open(custom_prompt_deactivate_script, 'w') as file:
            file.write(deactivate_script_content)
        print(f"Making the custom prompt deactivate script executable.")
        os.chmod(custom_prompt_deactivate_script, 0o755)  # Make the script executable

        print(f"Custom prompt setup complete for environment: {env_name}")
    except Exception as e:
        print(f"Error creating custom prompt scripts: {e}", file=sys.stderr)
        sys.exit(1)


def main_install():
    env_name = 'FPCAnalysisenv'
    python_version = '3.11'  # Specify the Python version you want here
    check_python_version(python_version)
    install_required_libraries(env_name)

if __name__ == "__main__":
    print("If this has any error, please see the comments at the bottom of install.py for debugging help!")
    time.sleep(3)

    main_install()

    if not check_latex_installed():
        print();print();print();print();
        print();print();print();print();
        print("LaTeX is required for this script to work with matplotlib.")
        print("Please install LaTeX AFTER LOADING CONDA ENV. For example, on Ubuntu you can run:")
        print("  sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended")
        print("Additionally, you may need (matplotlib > 3.2.1)")
        print("   sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super")
        print("Or for older")
        print("    sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended")
        print();print();print();print();
        print();print();print();print();

        time.sleep(10)

    print();print();print();print();
    print("Completed! Installation")
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help")
    print("Please activate the new enviroment! 'conda activate /path/to/here/FPCAnalysisenv' for linux/mac")
    print("This will need to be done every time a new terminal is launched!!!")


    confirmation = input("Would you like to modify the prompt script (by default, after activating the environment, you will see something like '(/full/path/to/install/FPCAnalysisenv)username@machinename:~/current/directory:' but afterwards you will see '(FPC)FPCAnalysis:' when the environment is activated). (Warning, current the script can not find the original prompt string and thus a terminal restart is required to return to the original prompt string after deactivating (this is annoying but harmless)!) Would you like the script to add a configuration file to your environment to do this?[y/n]: ")
    if confirmation.lower() == 'y':
        env_name = 'FPC'
        print(f"Okay! Starting the setup for custom prompt in environment: {env_name}")
        env_path = get_conda_env_path(env_name)
        create_custom_prompt_scripts(env_path, env_name)
        print(f"Setup complete.")
    else:
        print("Okay! Not modifying env config files/ prompt script..")


    print('*');print('*');print('*');print('*');
    print("Done with install... Please run `conda actiavte /path/to/here/FPCAnalysisenv' to activate the library or add '#!/path/to/here/FPCAnalysisenv/bin/python' (if on linux/mac) to the top of all scripts (and run by calling ./*scriptname.py)!")
    print("Optionally, use 'conda config --append envs_dirs /path/to/repo/FPCAnalysis/' (give the path to the parent 'FPCAnalysis', NOT the child/subfolder) to add the alias FPCAnalysisenv to conda for this directory. Then, you can call 'conda activate FPCAnalysisenv' to activate this environment")
    print("Use 'conda config --remove envs_dirs /path/to/repo/FPCAnalysis/' to remove this alias to uninstall or reinstall at a different location")
    print("When done, use conda deactivate to turn off the environment,")
    print("If using the environment, you will need to reactivate it every time you open a new terminal if you want to use the FPCAnalysis lib.")
    print("Be sure to activate the environment before launching a new jupyter notebook! Also, if you move the FPCAnalysisenv or FPCAnalysis folder, you may need to delete the FPCAnalysisenv folder and reinstall.")
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
