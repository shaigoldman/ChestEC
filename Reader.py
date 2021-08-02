import numpy as np
import pandas as pd
import scipy.io as sio

from cmlreaders import CMLReader, get_data_index
from th_eventreader import TH_EventReader

from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ResampleFilter
from ptsa.data.timeseries import TimeSeries


def subject_id(subject, montage, **kwargs):
    return (subject if montage == 0
                else f"{subject}_{int(montage)}")


def get_nav_epochs(path):
    """ Given path data of 1 event, determine the start
        and end of the navigation epoch.
        
        Args:
            path (list): list of path datapoints each containing
                ['mstime', 'x', 'y', 'heading']
        
        Returns:
            base_start (int): start of the baseline data, in mstime.
            move_start (int): start of the navigation epoch, in msitme.
    """
    xs = [p['x'] for p in path]
    ys = [p['y'] for p in path]
    dirs = [p['heading'] for p in path]
    ts = [p['mstime'] for p in path]

    started = False
    move_start = ts[-1]
    move_end = ts[-1]
    for i, (x,y,d,t) in enumerate(zip(xs, ys, dirs, ts)):
        if ((i>0) and
            ((x!=xs[i-1])
             or (y!=ys[i-1])
             or (d!=dirs[i-1])
                )):
            move_start = t
            started = True
        elif started:
            move_end = t
            break
    
    return move_start, move_end


def get_all_events(**subject_dict):
    
    df = get_data_index('r1')
    df = df[(df['experiment']==subject_dict['experiment'])
                 & (df['subject']==subject_dict['subject'])
                 & (df['montage']==subject_dict['montage'])
                ]
    original_sessions = df['original_session'].values
    sessions = df['session'].values

    all_events = []
    try:
        for session in sessions:
            all_events.append(TH_EventReader.get_events(
                subj=subject_dict['subject'], montage=subject_dict['montage'],
                session=session, exp=subject_dict['experiment']))
            
    except FileNotFoundError:
        for session in original_sessions:
            all_events.append(TH_EventReader.get_events(
                subj=subject_dict['subject'], montage=subject_dict['montage'],
                session=session, exp=subject_dict['experiment']))
            
    all_events = pd.concat(all_events)
    all_events.index = range(len(all_events))
    
    all_events = all_events[all_events['eegfile']!= '']
    
    if len(all_events['eegfile'].unique())>len(all_events['session'].unique()):
        # sometimes the eegfile splits in the middle of a session. This makes it
        # difficult to get data from the start of the first event with the new 
        # eegfile since part of the data will be at the end of the previous eegfile.
        # the best way to deal with it is to remove those events.
        bad_events = []
        for iloc, (i, row) in enumerate(all_events[['eegfile', 'session']].iterrows()):
            if row['eegfile'] != all_events.iloc[iloc-1]['eegfile']:
                if row['session'] == all_events.iloc[iloc-1]['session']:
                    bad_events.append(i)
        all_events = all_events[~all_events.index.isin(bad_events)]
    move_starts = []
    move_ends = []
    for i, event in all_events.iterrows():
        move_start, move_end = get_nav_epochs(event['pathInfo'])
        move_starts.append(move_start)
        move_ends.append(move_end)
    all_events['move_start'] = move_starts
    all_events['move_ends'] = move_ends
    
    return all_events


def load_matlab_contacts(subject, montage, load_type='contacts'):
    """ This is more consistent than cmlreaders version which breaks
        sometimes for no reason. Its also a lot more obvious what
        is going on whereas with cmlreaders its kinda a mystery :)
    """
    subj_str = subject_id(subject, montage)
    if load_type=='contacts':
        load_type = 'monopol'
        struct = 'talStruct'
    elif load_type=='pairs':
        load_type='bipol'
        struct = 'bpTalStruct'
    else:
        raise ValueError(f"load type must be 'pairs' or 'contacts': {load} is invalid load_type.'")
        
    path = f'/data/eeg/{subj_str}/tal/{subj_str}_talLocs_database_{load_type}.mat'
    contacts = pd.DataFrame(sio.loadmat(path, squeeze_me=True)[struct])
    
    if subject=='R1059J' and montage==1:
        # this contact is a duplicate of contact 114 
        contacts = contacts[contacts['channel'] != 113]
        contacts.index = range(len(contacts))
    
    return contacts


def load_contacts(subject_dict, load_type='contacts', **kwargs):
    
    reader = CMLReader(**subject_dict)
    contacts = reader.load(load_type)
    
    matlab_contacts = load_matlab_contacts(
        subject_dict['subject'], subject_dict['montage'],
        load_type=load_type,
    )
    for i in range(1,6):
        contacts[f'Loc{i}'] = matlab_contacts[f'Loc{i}']
    
    if not 'stein.region' in contacts:
        contacts['stein.region'] = np.nan
        
        if 'locTag' in matlab_contacts:
            loctags =  np.array([t if len(t)>0 else np.nan
                        for t in matlab_contacts['locTag']])
            contacts['stein.region'] = loctags

    contacts['stein.hemi'] = [np.nan if pd.isnull(t) or t == 'nan'
                              else t.split(' ')[0]
                              for t in contacts['stein.region']]
    contacts['stein.name'] = [np.nan if pd.isnull(t) or t == 'nan'
                                else ' '.join(t.split(' ')[1:])
                                for t in contacts['stein.region']]
    
    return contacts


def get_region_contacts(region, rcalc, contacts):
    if 'contact' in contacts:
        return contacts[contacts[rcalc]==region]['contact'].values
    else:
        return contacts[contacts[rcalc]==region]['contact_1'].values


def make_events_first_dim(ts, event_dim_str='event'):
    """
    Transposes a TimeSeries object to have the events dimension first. Returns transposed object.
    From jfmiller's miller_ecog_tools.
    """

    # if events is already the first dim, do nothing
    if ts.dims[0] == event_dim_str:
        return ts

    # make sure events is the first dim because I think it is better that way
    ev_dim = np.where(np.array(ts.dims) == event_dim_str)[0]
    new_dim_order = np.hstack([ev_dim, np.setdiff1d(range(ts.ndim), ev_dim)])
    ts = ts.transpose(*np.array(ts.dims)[new_dim_order])
    return ts


def load_eeg(subject_dict, events, which_contacts, 
             rel_start_ms, rel_stop_ms, buf_ms,
             noise_freq=[58., 62.], resample_freq=None,
             pass_band=None, do_average_ref=False,
             load_type='contacts',
             **kwargs,
            ):
    
    if buf_ms is not None:
        start = rel_start_ms - buf_ms
        stop = rel_stop_ms + buf_ms
    
    reader = CMLReader(**subject_dict)
    elec_scheme = load_contacts(subject_dict=subject_dict, load_type=load_type)
    contact_field = 'contact' if load_type=='contacts' else 'contact_1'
    
    loaded = False
    while not loaded:
        try:
            eeg = reader.load_eeg(
                events, scheme=elec_scheme,
                rel_start=start, rel_stop=stop, 
            ).to_ptsa()
            loaded=True
        except KeyError as ke:
            bad_contact = int(str(ke).replace("'", ''))
            if bad_contact in which_contacts:
                raise ke
            elec_scheme = elec_scheme[elec_scheme[contact_field] != bad_contact]
    
    # now auto cast to float32 to help with memory issues with high sample rate data
    eeg.data = eeg.data.astype('float32')

    if do_average_ref:
        # compute average reference by subracting the mean across channels
        eeg = eeg - eeg.mean(dim='channel')
    
    # reorder dims
    eeg = make_events_first_dim(eeg)
    
    # filter channels to only desired contacts
    contact_locs = [elec_scheme[elec_scheme[contact_field]==c].iloc[0].name
                    for c in which_contacts]
    eeg = eeg[:,  contact_locs, :]
        
    # filter line noise
    if noise_freq is not None:
        b_filter = ButterworthFilter(eeg, noise_freq, filt_type='stop', order=4)
        filt = b_filter.filter()
        # I'm not sure why, but for subj R1241J this is necessary:
        filt['event'] = eeg['event']
        eeg[:] = filt

    # resample if desired. Note: can be a bit slow especially if have a lot of eeg data
    if resample_freq is not None:
        eeg_resamp = []
        for this_chan in range(eeg.shape[1]):
            r_filter = ResampleFilter(eeg[:, this_chan:this_chan + 1], resample_freq)
            eeg_resamp.append(r_filter.filter())
        coords = {x: eeg[x] for x in eeg.coords.keys()}
        coords['time'] = eeg_resamp[0]['time']
        coords['samplerate'] = resample_freq
        dims = eeg.dims
        eeg = TimeSeries.create(np.concatenate(eeg_resamp, axis=1),
                                resample_freq, coords=coords,
                                dims=dims)

    # do band pass if desired.
    if pass_band is not None:
        eeg = ButterworthFilter(eeg, pass_band, filt_type='pass', order=4).filter()
        
    eeg = make_events_first_dim(eeg)
    
    return eeg
