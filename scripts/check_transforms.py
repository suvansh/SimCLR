import os
import sys
import pickle
from utils import get_repo_dir

if len(sys.argv) > 1:
    check_type = sys.argv[2]
    assert check_type in ['partial', 'full']
else:
    check_type = 'partial'

def result(passed):
    word = 'PASSED' if passed else 'FAILED'
    print(f'Test {word}! Ran on {len(dir_names)} directories.')

transform_dir = os.path.join(get_repo_dir(), 'data/transforms')
dir_names = set(f.name for f in os.scandir(transform_dir) if f.is_dir())
hashes = pickle.load(open(os.path.join(transform_dir, 'hashes.pkl'), 'rb'))

if check_type == 'partial':
    result(dir_names.issubset(hashes))
else:
    result(dir_names == hashes)
