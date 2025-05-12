'''
# ----------------------
#  interactive_session
# ----------------------

'''

# == imports ==
# -- packages --
import os
import sys
import importlib
import subprocess

# -- util- and local scripts --
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, file_path):
    ''' import module from relative path '''
    if file_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        cwd = os.getcwd()
        relative_path = os.path.relpath(file_path, cwd) # ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                   'utils')


# == start session ==
def start_interactive_session():
    output = mS.get_user_specs()
    folder_work, folder_scratch, SU_project, storage_project, data_projects = output
    storage_string = f"storage={'+'.join([f'gdata/{project}' for project in data_projects])}+scratch/{storage_project}"
    try:
        command = [
            "qsub",
            "-I",                                                                                                                   # Request interactive session
            f"-P", SU_project,                                                                                                      # Project
            "-l", storage_string,
            "-l", "wd",                                                                                                             # Working directory
            "-q", "normal",                                                                                                         # Queue
            "-l", "walltime=2:00:00",                                                                                               # Wall time
            "-l", "mem=50GB",                                                                                                       # Memory
            "-l", "ncpus=1",                                                                                                        # CPUs
            "-l", "jobfs=200GB"                                                                                                     # Job file system
        ]
        subprocess.run(command, check=True)
        print("Interactive session completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting interactive session: {e}")
    except KeyboardInterrupt:
        print("Interactive session cancelled.")
    finally:
        subprocess.run(["rm", "-f", "interactive_session.pbs"])

# == when this script is ran ==
if __name__ == "__main__":
    start_interactive_session()

