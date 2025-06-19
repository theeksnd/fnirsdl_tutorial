#create the dataset for model traning
import numpy as np
import cedalion
import numpy as np 
from cedalion.io import read_events_from_tsv
import xarray as xr
from pathlib import PureWindowsPath
from cedalion.io import read_snirf
import glob
import os
import pickle
import cedalion.sigproc.motion_correct as motion_correct

# to RUN: apptainer run --nv --bind `pwd`/xkb:/var/lib/xkb,`pwd`/cedalion:/app cedalion.sif python LOCATION/preprocessfinal.py 

DIR = "./dataset_v3" # location of the dataset
try:
    os.mkdir(DIR)
except FileExistsError:
    pass

INDEX = 0
FMAX = 0.7 #for augmentation, use 0.5 Hz for default

DIR = '{}/frq{}'.format(DIR, FMAX)
try:
    os.mkdir(DIR)
except FileExistsError:
    pass

subject_to_rec = {}
for file in glob.glob("./BallSqueezingHD/sub*/nirs/sub*.snirf"): #location of the snirf files, use the regular expression to match the files
    #read file
    try:
        recs = read_snirf(file)
    except OSError:
        print('failed to read', file)
        continue

    print('processing:', file)
    rec = recs[0]

    #read-the stimulus
    stim = read_events_from_tsv(file.replace('nirs.snirf', 'events.tsv'))
    label = {'Right':1, 'Left':2}

    amp = rec['amp']
    #amplitude data
    amp = amp.pint.dequantify().pint.quantify("V")
    #montage informaiton
    dpf = xr.DataArray([6., 6.], dims="wavelength", coords={"wavelength" :  rec["amp"].wavelength})
    #BL-transform
    rec["od"], baseline = cedalion.nirs.int2od(rec["amp"], return_baseline=True)
    # compute motion correct
    rec["od_tddr"] = motion_correct.tddr(rec["od"])
    rec["od_wavelet"] = motion_correct.motion_correct_wavelet(rec["od_tddr"])

    # final amplitude of the data
    rec["amp_final"] = cedalion.nirs.od2int(rec["od_wavelet"], baseline)
    rec['blt'] = cedalion.nirs.beer_lambert(rec["amp_final"], rec.geo3d, dpf)
    #freq-filter
    rec['freq-f'] = rec['blt'].cd.freq_filter(fmin=0.01, fmax=FMAX, butter_order=4)

    # baseline correction
    baseline_conc = rec['freq-f'].mean("time")
    rec["final"] = rec['freq-f'] - baseline_conc

    #extrat meta information
    data = rec['final'].as_numpy()
    time = rec['final'].time.as_numpy()

    reference = np.zeros(time.shape[0])
    SUB = PureWindowsPath(file).parts[-3]
    if SUB not in subject_to_rec:
        subject_to_rec[SUB] = []
    try:
        os.mkdir('{}/{}'.format(DIR, SUB))
    except FileExistsError:
        pass
    delta = 2.5
    for onset, du, ty in zip(stim['onset'], stim['duration'], stim['trial_type']):
        #update the reference with correct labels
        idx = np.where((time >= onset) & (time < onset+du))[0]
        reference[idx] = label[ty]
        #shift the data to generate more samples

        for delta_ in np.linspace(-delta, delta, 9):
            idx_sh = np.where((time >= onset+delta_) & (time < onset+du+delta_))[0]
            segment_sh = data[:, :, idx_sh]
            ref_segment = reference[idx_sh]
            if segment_sh.shape[-1] != 87:
                continue
            #save files
            filename = r'{}/{}/event_{}_{}.pkl'.format(DIR, SUB, 'aug' if delta_ != 0.0 else '', INDEX)
            subject_to_rec[SUB].append(filename)
            INDEX += 1
            print(filename)
            # dataset information            
            segment_data = {
                'xt': segment_sh[:, :, :87], #we manually adjust for time impurities
                'ref': ref_segment[:87],
                'class': label[ty],
                'file': file
            }
            # write
            with open(filename, 'wb') as handle:
                pickle.dump(segment_data, handle, protocol=pickle.HIGHEST_PROTOCOL)     
with open('{}/meta_event_{}.pkl'.format(DIR, FMAX), 'wb') as handle:
    pickle.dump(subject_to_rec, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('# saved:', INDEX)
