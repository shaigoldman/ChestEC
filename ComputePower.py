import pandas as pd
import numpy as np
import numexpr
import bottleneck as bn

import scipy.signal # to use signal.hilbert
import scipy.fftpack # to use fftpack.next_fast_len
from scipy.stats.mstats import zscore

from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ButterworthFilter

import Reader


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


def compute_wavelet(eeg, freqs, wave_num=5,
                    output='power', log_power=True,
                    mean_over_freqs=False,
                   ):

    wave_pow = MorletWaveletFilter(
        eeg, freqs, output=output, width=wave_num,
        cpus=12, verbose=False).filter()
    
    if output=='power' and log_power:
        data = wave_pow.data
        wave_pow.data = numexpr.evaluate('log10(data)')
       
    if mean_over_freqs:
        wave_pow = wave_pow.mean(dim='frequency')
    
    return wave_pow


def get_time_bins(window_size, window_step, start, end):
    
    time_bins = [[i, i+window_size]
                 for i in np.arange(start,
                                    end+window_step,
                                    window_step)
                ]
    time_bins = pd.DataFrame(
        time_bins, columns=['start', 'end'],
        index=range(len(time_bins))
    )
    return time_bins


def compute_power(eeg, freq_band, buf_ms,
                  window_size=None, window_step=100,
                  transform='hilbert',
                  **wave_kwargs):
    
    if transform == 'hilbert':
        filt_eeg = ButterworthFilter(
            eeg, freq_band, filt_type='pass', order=4
        ).filter()
        powers = compute_hilbert(filt_eeg)
        
    elif transform == 'wavelet':
        
        num_freqs = int(freq_band[1]-freq_band[0]) * 2
        
        freqs = np.geomspace(*freq_band, num_freqs)
        powers = compute_wavelet(
            eeg, freqs, mean_over_freqs=True,
            **wave_kwargs
        )

    else:
        raise ValueError(f'Transform {transform} is invalid. Pick from ["hilbert", "wavelet"]')
    
    if buf_ms:
        powers = powers.remove_buffer(buf_ms/1000.)
    

    # take the mean of each time bin, if given
    # create a new timeseries for each bin and the
    # concat and add in new time dimension. This code
    # fragment is from jfmiller.
    if window_size is not None:

        time_bins = get_time_bins(window_size, window_step,
                                  start=powers.time.data[0],
                                  end=powers.time.data[-1]
                                 )

        # figure out window size based on sample rate\
        window_size = int(window_size * powers.samplerate.data / 1000.)

        # compute moving average with window size that we want to average over (in samples)
        pow_move_mean = bn.move_mean(powers.data, window=window_size, axis=2)

        # reduce to just windows that are centered on the times we want
        powers.data = pow_move_mean
        powers = powers[:, :, np.searchsorted(powers.time.data, time_bins.iloc[:, 1].values) - 1]

        # set the times bins to be the new times bins (ie, the center of the bins)
        powers['time'] = time_bins.mean(axis=1).values
    
    return powers


def get_basepow(events, freq_band, which_contacts, buf_ms,
                subject_dict, eeg_kwargs={}, pow_kwargs={}, wave_kwargs={}):
    """ Get pre-trial baseline powers.
    
        Args:
            events (pd.DataFrame)
            freq_band (list): two floats.
            which_contacts (list of ints)
            buf_ms (int or float)
            subject_dict (dict):
                ['subject', 'montage', 'experiment', 'localization']
            eeg_kwargs (dict):
                [noise_freq=[58., 62.], resample_freq=None]
            pow_kwargs (dict):
                [window_size=None, window_step=100, transform='hilbert']
            wave_kwargs (dict):
                [wave_num=6, log_power=True]
            
    """
    
    sessions = events['session'].drop_duplicates()
    
    besepow_mu = [[] for sess in sessions]
    basepow_std = [[] for sess in sessions]
    
    for i, (s, session) in enumerate(sessions.iteritems()):
    
        sess_trials = events[
            events['session']==session]['trial'].drop_duplicates()
        for e, trial in sess_trials.iteritems():
            # ^ this will give e as the index of the first event from each trial
            event = events.loc[e]
            
            # determine baseline starts and stops
            rel_start_ms = event['base_start'] - event['mstime']
            rel_stop_ms = event['move_start'] - event['base_start']
            
            # load baseline eeg and power
            pre_trial_eeg = Reader.load_eeg(
                events.loc[e:e], which_contacts=which_contacts,
                rel_start_ms=rel_start_ms, rel_stop_ms=rel_stop_ms,
                buf_ms=buf_ms, do_average_ref=False,
                **subject_dict, **eeg_kwargs,
            )
            pre_trial_power = compute_power(
                pre_trial_eeg, freq_band, buf_ms=buf_ms,
                **pow_kwargs, **wave_kwargs

            )
            
            # get mean and std for each channel
            basepow_mu[i].append(pre_trial_power.mean(dim=['event', 'time']))
            basepow_std[i].append(pre_trial_power.std(dim=['event', 'time']))
    
    # convert to xarrays
    basepow = xr.DataArray([basepow_mu, basepow_std],
                           coords=[['mu', 'std'], sessions, pre_trial_power.channel],
                           dims=['zstat', 'session', 'channel'])

    return basebasepowpow


def zscore_powers(powers, basepow):
    
    powers = powers.copy() # dont want to change anything in place
    
    # z score each session based on its own baseline
    sessions = powers.event.values['session'].unique()
    for session in sessions:
        sess_locs = powers.event.values['session']==sessions
        # zscore will work accross channels, for all events in this session
        powers[sess_locs] -= basepow['mu', session]
        powers[sess_locs] /= basepow['std', session]

    return powers
