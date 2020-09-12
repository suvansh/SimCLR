import os
import sys
import shutil
from utils import get_repo_dir
from tqdm import tqdm


trans_dir = os.path.join(get_repo_dir(), 'data/transforms')
to_delete = []
for hash_dir in tqdm([f for f in os.scandir(trans_dir) if f.is_dir()]):
    if len(list(os.scandir(hash_dir))) != 200:
       to_delete.append(hash_dir) 
if to_delete:
    if (len(sys.argv) > 1 and sys.argv[1] == 'DELETE'
        and input(f'Are you sure you want to delete {len(to_delete)} directories? (y/N): ').lower() in ['y', 'yes']):
        for hash_dir in to_delete:
            print(f'Deleting directory {hash_dir.name}')
            shutil.rmtree(hash_dir.path)
    else:
        print(f'{len(to_delete)} incomplete directories found.')
        if input('Print? (y/N): ').lower() in ['y', 'yes']:
            for hash_dir in to_delete:
                print(hash_dir.name)

else:   
    print('No incomplete directories found.')
