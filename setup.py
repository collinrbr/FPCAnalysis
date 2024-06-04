#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil

def check_python_version(version):
    python_executable = f"python{version}"  # Modify the prefix as needed
    if os.system(f"{python_executable} --version > /dev/null 2>&1") != 0:
        print(f"Python {version} is not installed.")
        print("Please install python3.11! (Try 'sudo apt-get install python3.11' for linux and 'brew install python@3.11' for mac (requires brew!)'")
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

    subprocess.run([pip_executable, 'install', '--no-cache-dir', '-r', requirements_file])

    # Determine the paths for the conda environment
    if os.name == 'nt':  # Windows
        conda_activate = os.path.join(env_name, 'Scripts', 'activate')
    else:  # Unix/Linux/MacOS
        conda_activate = os.path.join(env_name, 'bin', 'activate')

    #install postgkeyll
    print("Installing postgkeyll (Warning: we install a specific version- not the most current version!)")
    os.chdir("..")
    os.system('git clone https://github.com/ammarhakim/postgkyl.git')
    os.chdir("postgkyl")
    os.system('git checkout f876908d9e0969e7608100c7b769410e21c56549') #'f876908' is the specific commit we are debugging with. (see https://github.com/ammarhakim/postgkyl/tree/f876908d9e0969e7608100c7b769410e21c56549)
                                                                       #Update this line (be sure the packages in requirements.txt satisfy the new requirements for postgkyl) to the latest commit to update!
    os.chdir("..")
    os.system('mv postgkyl FPCAnalysis')
    os.chdir("FPCAnalysis")
    subprocess.run([pip_executable, 'install', '-e', 'postgkyl'])
    print("Done!")

    # Create a directory for your package within the site-packages directory
    site_packages_dir = os.path.join(env_name, 'lib', 'python3.11', 'site-packages')
    package_dir = os.path.join(site_packages_dir, 'fpcanalysis')  # Replace 'your_package_name' with your desired package name

    if not os.path.exists(package_dir):
        os.makedirs(package_dir)

    # Copy all .py files from your library directory to the package directory
    library_dir = 'lib'  #path to fpc analysis lib in main repo
    for file_name in os.listdir(library_dir):
        if file_name.endswith('.py'):
            shutil.copy(os.path.join(library_dir, file_name), package_dir)
        elif file_name == 'plot':
            print("debug: ",os.path.join(library_dir, file_name),package_dir)
            shutil.copytree(os.path.join(library_dir, file_name),package_dir+'/plot')

    # # Activate the conda environment
    # try:
    #     if os.name == 'nt':  # Windows
    #         activate_script = os.path.join(env_name, 'Scripts', 'activate')
    #         subprocess.run([activate_script], shell=True, check=True)
    #     else:  # Unix/Linux/MacOS
    #         activate_script = os.path.join(env_name, 'bin', 'activate')
    #         subprocess.run(['source '+str(activate_script)], shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"An error occurred while activating the environment: {e}")
    #     return
        
    # # Install ADIOS2 using conda
    # print("Installing ADIOS2 using conda-forge")
    # try:
    #     subprocess.run(['conda', 'install', '-c', 'conda-forge', 'adios2>=2.9.2', '--verbose', '-y'], check=True)
    #     print("ADIOS2 installed successfully!")
    # except subprocess.CalledProcessError as e:
    #     print(f"An error occurred while installing ADIOS2: {e}")

    


def main():
    env_name = 'FPCAnalysis'
    python_version = '3.11'  # Specify the Python version you want here
    check_python_version(python_version)
    create_virtualenv(env_name, python_version)
    install_required_libraries(env_name)

if __name__ == "__main__":
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help!")

    import time
    time.sleep(5)

    main()
    print("Completed! Installation")
    print("If this has any error, please see the comments at the bottom of setup.py for debugging help")
    print("Please activate the new enviroment! 'source FPCAnalysis/bin/activate' for linux/mac and 'FPCAnalysis\\Scripts\\activate' for windows")
    print("This will need to be done every time a new terminal is launched!!!")

    print("Attempting to run first import to compile binaries...")
    try:
        os.system('touch _tempimport.py')
        os.system('chmod 777 _tempimport.py')
        os.system("echo '#!FPCAnalysis/bin/python' >> _tempimport.py")
        os.system("echo 'import fpcanalysis' >> _tempimport.py")
        os.system('./_tempimport.py')
        os.system('rm _tempimport.py')
        print("Done!")
    except:
        print("Was not able to import for the first time to compile binaries. First run with library may be slower than usual as binaries will need to compile!")

    #General fixes!
    # Always a good idea to update pip 'python -m pip install --upgrade pip'
    #     and to update conda with 'conda update conda'

    #Ubuntu fixes!
    #If you get "FileNotFoundError: [Errno 2] No such file or directory: 'python3.11'" try running:
    #sudo apt-get install python3.11
    
    #If you get "ModuleNotFoundError: No module named 'distutils.util'" try running: 
    #sudo apt-get install --reinstall python3.11-distutils
