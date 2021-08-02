import pandas as pd
import numpy as np
import xarray as xr

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from ptsa.data.timeseries import TimeSeries
from cmlreaders import CMLReader, get_data_index

import Reader
import ComputePower

# default variable values
EVENTS_TYPE = 'CHEST'
REL_START=-2250
REL_STOP=3750
BUFFER=3000
WINDOW_SIZE=500
TRANSFORM='wavelet'
FREQ_BAND=[1,3]
RESAMPLE_FREQ = 500


def get_erps_1subj(subject_dict, region, 
                   freq_band=FREQ_BAND, events_type=EVENTS_TYPE, 
                   rel_start_ms=REL_START, rel_stop_ms=REL_STOP,
                   transform=TRANSFORM, window_size=WINDOW_SIZE,
                   buf_ms=BUFFER, resample_freq=None,
                   **kwargs
                  ):
    
    # get events
    events = Reader.get_all_events(**subject_dict)
    events = events[events['type']==events_type]
    events.index = range(len(events))
    
    # get elecs
    all_elecs = Reader.load_contacts(subject_dict=subject_dict, **kwargs)
    contacts = Reader.get_region_contacts(region, 'stein.region', all_elecs)
    if len(contacts)==0:
        raise ValueError(f"subject {subject_dict} has no electrodes in region {region}")
    
    # get baseline power
    basepow = ComputePower.get_basepow(
        events=events, freq_band=freq_band,
        which_contacts=contacts, buf_ms=buf_ms,
        subject_dict=subject_dict,
        resample_freq=resample_freq,
        transform=transform,
        **kwargs)
    
    # get powers
    eeg = Reader.load_eeg(subject_dict, events, contacts, 
                          rel_start_ms=REL_START, rel_stop_ms=REL_STOP,
                          do_average_ref=False, 
                          buf_ms=buf_ms, resample_freq=resample_freq,
                          **kwargs,
                  )
    
    powers = ComputePower.compute_power(eeg, freq_band=freq_band, buf_ms=buf_ms,
                                        window_size=window_size, transform=transform,
                                        **kwargs)
    
    
    powers = ComputePower.zscore_powers(powers, basepow)
    #powers = ComputePower.zscore_old(powers)
    
    # split by recalled, not recalled, and empty chests
    recalled_locs = events['recalled'].values
    empty_locs = (events['item_name']=='').values
    not_recalled_locs = ~(recalled_locs|empty_locs)
    
    # get powers accross events
    rec_power = powers[recalled_locs].mean(dim='event')
    nrec_power = powers[not_recalled_locs].mean(dim='event')
    empty_power = powers[empty_locs].mean(dim='event')
    
    # get erps accross channels
    rec_erp = rec_power.mean(dim='channel')
    rec_sem = rec_power.std(dim='channel') / np.sqrt(rec_power.channel.size)

    nrec_erp = nrec_power.mean(dim='channel')
    nrec_sem = nrec_power.std(dim='channel') / np.sqrt(nrec_power.channel.size)

    empty_erp = empty_power.mean(dim='channel')
    empty_sem = empty_power.std(dim='channel') / np.sqrt(nrec_power.channel.size)

    # done
    return pd.DataFrame({'rec_erp': rec_erp, 'rec_sem': rec_sem,
                         'nrec_erp': nrec_erp, 'nrec_sem': nrec_sem,
                         'empty_erp': empty_erp, 'empty_sem': empty_sem, 
                         'samplerate': rec_power.samplerate.values[()],
                         'window_size': window_size,
                         'freq_band': str(freq_band)
                        }, index=rec_erp.time)


def get_erps(subjects, region, freq_band,
             graphit=False, resample_freq=RESAMPLE_FREQ,
             window_size=WINDOW_SIZE, **kwargs
            ):

    if graphit:
        # set up graph on 1 subplot per subj
        ncols = int(np.sqrt(len(subjects)))
        nrows = int(np.ceil(len(subjects)/ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,15))
        fig.tight_layout()

    # store each subjects erps in these lists
    all_recs = []
    all_nrecs = []
    all_emptys = []
    
    r_sems = []
    nr_sems = []
    empt_sems = []
    
    subjects_with_data = []

    # get each subject's erp
    for i, (ind, subject) in tqdm(enumerate(subjects.iterrows()), total=len(subjects)):

        try:
            erps = get_erps_1subj(
                subject_dict=subject,
                region=region,
                freq_band=freq_band,
                resample_freq=resample_freq,
                window_size=window_size,
                **kwargs
            )

            all_recs.append(erps['rec_erp'])
            all_nrecs.append(erps['nrec_erp'])
            all_emptys.append(erps['empty_erp'])
            
            r_sems.append(erps['rec_sem'])
            nr_sems.append(erps['nrec_sem'])
            empt_sems.append(erps['empty_sem'])
            
            subjects_with_data.append(
                Reader.subject_id(**subject))

            if graphit:
                ax = axes[i%nrows, int(i/nrows)]
                plot_erp(erps, axes=ax, do_legend=False)
                ax.set_title(Reader.subject_id(**subject))
        except (IndexError, ValueError) as e:
            continue
            
    if graphit:
        plt.show()
    
    # concatenate erps accross subjects 
    recs = pd.concat(all_recs, axis=1)
    r_sems = pd.concat(r_sems, axis=1)
    recs = xr.DataArray([recs.values.transpose(), r_sems.values.transpose()],
                        coords=[['erp', 'sem'], subjects_with_data, recs.index],
                        dims = ['stat', 'subject', 'time'])

    nrecs = pd.concat(all_nrecs, axis=1)
    nr_sems = pd.concat(nr_sems, axis=1)
    nrecs = xr.DataArray([nrecs.values.transpose(), nr_sems.values.transpose()],
                        coords=[['erp', 'sem'], subjects_with_data, nrecs.index],
                        dims = ['stat', 'subject', 'time'])

    emptys = pd.concat(all_emptys, axis=1)
    empt_sems = pd.concat(empt_sems, axis=1)
    emptys = xr.DataArray([emptys.values.transpose(), empt_sems.values.transpose()],
                        coords=[['erp', 'sem'], subjects_with_data, emptys.index],
                        dims = ['stat', 'subject', 'time'])
    
    # get erps accross subjects
    
    # Note: I have two forms of sem calc, one is commented out.
    # the commented out one is accross subjects, which feels more logical
    # to me conceptually, but looks bad, and the one I'm using is within-subj
    # which is I think what jmiller used in his paper for fig 5
    rec_erp = recs[recs.stat=='erp'].mean(dim='subject').squeeze()
    rec_sem = recs[recs.stat=='sem'].mean(dim='subject').squeeze()
    #rec_sem = recs[recs.stat=='erp'].std(dim='subject').squeeze() / np.sqrt(recs.subject.size)

    nrec_erp = nrecs[nrecs.stat=='erp'].mean(dim='subject').squeeze()
    nrec_sem = nrecs[nrecs.stat=='sem'].mean(dim='subject').squeeze()
    #nrec_sem = nrecs[nrecs.stat=='erp'].std(dim='subject').squeeze() / np.sqrt(nrecs.subject.size)

    empty_erp = emptys[emptys.stat=='erp'].mean(dim='subject').squeeze()
    empty_sem = emptys[emptys.stat=='sem'].mean(dim='subject').squeeze()
    #empty_sem = emptys[emptys.stat=='erp'].std(dim='subject').squeeze() / np.sqrt(emptys.subject.size)
    
    time_index = rec_erp.time

    # done
    return pd.DataFrame({'rec_erp': rec_erp, 'rec_sem': rec_sem,
                         'nrec_erp': nrec_erp, 'nrec_sem': nrec_sem,
                         'empty_erp': empty_erp, 'empty_sem': empty_sem, 
                         'samplerate': resample_freq, 'window_size': window_size,
                         'freq_band': f'{freq_band[0]}-{freq_band[1]}'
                        }, index=time_index)


def error_fill(xs, ys, err, color, label, axes=None):
    
    if axes == None:
        plotter = plt
    else:
        plotter = axes
    
    plotter.fill_between(xs, ys-err, ys+err,
                     alpha=.4, color=color)
    plotter.plot(xs, ys, label=label, color=color)
    


def plot_erp(erps, axes=None, do_legend=True):
    
    samplerate = erps['samplerate'].iloc[0]
    freq_band = erps['freq_band'].iloc[0]
    window_size = erps['window_size'].iloc[0]
    
    xs = erps.index
    if window_size is not None:
        xs -= window_size/2
    xs /= 1000
    
    rec_color = 'firebrick'
    nrec_color = 'steelblue'
    empty_color = 'gray'
    
    if axes is None:
        axes = plt.subplot()

    error_fill(xs, erps['rec_erp'], erps['rec_sem'], color=rec_color, label='recalled', axes=axes)
    error_fill(xs, erps['nrec_erp'], erps['nrec_sem'], color=nrec_color, label='not recalled', axes=axes)
    error_fill(xs, erps['empty_erp'], erps['empty_sem'], color=empty_color, label='empty', axes=axes)

    axes.axvline(0, color='k')
    axes.axvline(1.5, color='k', linestyle=':') # item stops being displayed
    axes.axhline(0, color='k', linestyle='--')
    
    axes.set_xlabel('time (ms)')
    axes.set_ylabel(fr'Z(power) $\emdash$ [{freq_band}] Hz')

    if do_legend:
        axes.legend()
        
    axes.set_xlim((-2.5,4))