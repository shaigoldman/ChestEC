import pandas as pd
import numpy as np

import scipy.signal # to use signal.hilbert
import scipy.fftpack # to use fftpack.next_fast_len
from scipy.stats.mstats import zscore

from ptsa.data.timeseries import TimeSeries


def compute_hilbert(eeg):
    """ Computes power using hilbert transform. Based off code
        the youtube series I reference in the comments of the code.
    """
    # Using modified hilbert function for speed (see https://github.com/scipy/scipy/issues/6324)
    fast_hilbert = lambda x: scipy.signal.hilbert(
        x, scipy.fftpack.next_fast_len(x.shape[-1])
    )[:, :, :x.shape[-1]]
    # abs(hilbert) gives the amplitute, sqaure(amplitute) gives power.
    # see: https://www.youtube.com/watch?v=VyLU8hlhI-I&t=421s
    amp = np.abs(fast_hilbert(eeg))
    hilbert = np.square(amp)
    # ("TimeSeries" is a class from ptsa)
    hilbert_pow = TimeSeries(data=hilbert, 
                             coords=eeg.coords, 
                             dims=eeg.dims
                            )
    return hilbert_pow


def zscore_eeg(eeg_pow):
    
    z_pow = zscore(eeg_pow, axis=eeg_pow.get_axis_num('time'))
    z_pow = TimeSeries(data=z_pow, coords=eeg_pow.coords,
                       dims=eeg_pow.dims)  
    return z_pow


def compute_power(eeg, buf_ms):
    hilbert = compute_hilbert(eeg).remove_buffer(buf_ms/1000.)
    z_pow = zscore_eeg(hilbert)
    return z_pow