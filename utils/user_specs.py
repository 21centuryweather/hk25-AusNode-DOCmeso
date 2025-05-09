'''
# -----------------
#   user_specs
# -----------------
scratch:    
/scratch/k10/cb4968
work:       
/g/data/k10/cb4968

'''

# == imports ==
# -- packages --
import os


# == get user ==
def get_user_specs(show = False):
    username =          os.path.expanduser("~").split('/')[-1]                          # ex; 'cb4968'
    storage_project =   'k10'                                                           # storage project
    SU_project =        'gb02'                                                          # resource project
    data_projects =     ('hh5', 'al33', 'oi10', 'ia39', 'rt52', 'fs38', 'k10', 'gb02')  # directories available for job
    folder_scratch =    (f'/scratch/{storage_project}/{username}')                      # temp files
    folder_work =       (f'/g/data/{storage_project}/{username}')                       # saved
    if show:
        print('user specs:')
        [print(f) for f in [username, storage_project, SU_project, data_projects, folder_scratch, folder_work]]
    return folder_work, folder_scratch, SU_project, storage_project, data_projects


# == when this script is ran ==
if __name__ == '__main__':
    output = get_user_specs()
    folder_work, folder_scratch, SU_project, storage_project, data_projects = output
    [print(f) for f in output]






