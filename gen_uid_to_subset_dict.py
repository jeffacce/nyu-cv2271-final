from pathlib import Path
import json


uid_to_subset = {}
root = Path('/scratch/zc2357/cv/final/datasets/luna16')

for i in range(10):
    paths = list((root / ('subset%s' % i)).glob('*.mhd'))
    this_subset = {x.stem: x.parent.relative_to(root).as_posix() for x in paths}
    uid_to_subset.update(this_subset)


with open(root / 'uid_to_subset.json', 'w') as f:
    json.dump(uid_to_subset, f)
