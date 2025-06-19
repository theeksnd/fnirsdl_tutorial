import pickle

# this script is only for the BallSqueezingHD dataset
freqs = [0.5, 0.7, 1.0] # your frequency bands

# dataset location
events = "/home/theek/ibs_dl/dataset_v3/frq{}/meta_event_{}.pkl"

meta_events = []
for freq in freqs:
    with open(events.format(freq, freq), 'rb') as handle:
        meta = pickle.load(handle)
    meta_events.append(meta)

files_to_session = {}
session_to_files = {'run-1':[], 'run-2':[], 'run-3':[]}

for meta_event in meta_events:
    for sub in meta_event:
        for file in meta_event[sub]:
            meta = None
            with open(file, 'rb') as handle:
                meta = pickle.load(handle) 
            run = meta['file'].split('_')[-2]
            files_to_session[file] = run
            session_to_files[run].append(file)

for run in session_to_files:
    print(run, len(session_to_files[run]))

# this will save the mapping of files to sessions used for LOSO
with open('files_to_sessions_v3.pkl', 'wb') as handle:
    pickle.dump(files_to_session, handle, protocol=pickle.HIGHEST_PROTOCOL)     
