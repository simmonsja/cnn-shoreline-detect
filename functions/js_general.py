import sys, os
import numpy as np

def progress_bar(ii, iiTotal):
    sys.stdout.write('\r')
    percdone = (ii+1)/float(iiTotal)
    sys.stdout.write("[%-20s] %d%% - Processing: %d of %d." % ('='*int(np.floor(percdone*20)), percdone*100, ii+1, iiTotal))
    sys.stdout.flush()

# Disable
def blockPrint():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return old_stdout

# Restore
def enablePrint(existing=None):
    if existing:
        sys.stdout = existing
    else:
        sys.stdout = sys.__stdout__
