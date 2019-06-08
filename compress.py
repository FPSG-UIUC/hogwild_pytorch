#!/usr/bin/python
"""Iterate over generated runs and compress them"""
import os
from tqdm import tqdm


def get_directories():
    """Find and process directories

    Returns a list of directories matching the criteria"""
    cmd = os.popen(r"ls -l -h | rg -e 'Jun\s+7' -e 'Jun\s+6' | \
                   rg -e 'sim.*hogwild' | sed -e 's|^.* ||'")
    return cmd.read().splitlines()


FNAMES = ['conf.{}'.format(i) for i in range(10)]
FNAMES.append('eval')


def compress(dirs):
    """Iterate over files and compress them"""
    with tqdm(unit="Files", total=len(dirs)*11) as pbar:
        for cdir in dirs:
            for cfile in FNAMES:
                if not os.path.exists("{}/{}".format(cdir, cfile)):
                    cmd = "cd {}; gzip {}".format(cdir, cfile)
                    output = os.popen(cmd)
                    for line in output.read().splitlines():
                        print(line)
                pbar.update(1)


def copy(dirs):
    """Iterate over output directories and copy them to shared space"""
    for cdir in tqdm(dirs, unit="Output Directories", total=len(dirs)):
        cmd = "cp -r /scratch/{} /shared/jose/pytorch/outputs/".format(cdir)
        output = os.popen(cmd)
        for line in output.read().splitlines():
            print(line)
