import os
import re


def get_continuation_fname(logdir):

    import pdb; pdb.set_trace()

    # Take last
    files = [f for f in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, f))]

    if not files:
        assert False, f"{logdir} does not corresponding to an existing resumable model."
        
    epochNames = [re.match(f"epoch(\d+)_step\d+.*\.pt$", f) for f in files]
    epochs = [match.group(1) for match in filter(lambda x: x is not None, epochNames)]
    maxEpoch = sorted(epochs, reverse=True, key=int)[0]

    fPrefix = f'epoch{maxEpoch}_step'
    fnames = [re.match(f"({re.escape(fPrefix)}\d+).*\.pt$", f) for f in files]
    # Take first if multiple matches (unlikely)
    fname = ([f for f in fnames if f is not None][0]).group(1)

    epoch = int(maxEpoch)

    return fname, epoch
