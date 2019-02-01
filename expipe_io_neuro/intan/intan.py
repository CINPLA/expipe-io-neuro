import exdir
import shutil
import glob
import os
import quantities as pq
import numpy as np

# from expipe.core import Filerecord
# from expipe.core import user
# from expipe import settings
import os.path as op


def _prepare_exdir_file(exdir_file):
    general = exdir_file.require_group("general")
    subject = general.require_group("subject")
    processing = exdir_file.require_group("processing")
    epochs = exdir_file.require_group("epochs")

    return general, subject, processing, epochs


def convert(intan_file, exdir_path, session):
    exdir_file = exdir.File(exdir_path, plugins=exdir.plugins.quantities)
    dtime = intan_file.datetime.strftime('%Y-%m-%dT%H:%M:%S')
    exdir_file.attrs['session_start_time'] = dtime
    exdir_file.attrs['session_duration'] = intan_file.duration
    acquisition = exdir_file.require_group("acquisition")
    general = exdir_file.require_group("general")
    processing = exdir_file.require_group("processing")
    subject = general.require_group("subject")

    target_folder = op.join(str(acquisition.directory), session)
    acquisition.attrs["session"] = session
    acquisition.attrs["acquisition_system"] = intan_file.acquisition_system

    print("Copying ", intan_file.absolute_filename, " to ", target_folder)
    if not op.isdir(target_folder):
        os.mkdir(target_folder)
    shutil.copyfile(intan_file.absolute_filename, op.join(target_folder, intan_file.fname))


def _prepare_channel_groups(exdir_path, intan_file):
    exdir_file = exdir.File(exdir_path, plugins=exdir.plugins.quantities)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)

    exdir_channel_groups = []
    elphys = processing.require_group('electrophysiology')
    for intan_channel_group in intan_file.channel_groups:
        exdir_channel_group = elphys.require_group(
            "channel_group_{}".format(intan_channel_group.channel_group_id))
        exdir_channel_groups.append(exdir_channel_group)
        channel_identities = np.array([ch.index for ch in intan_channel_group.channels])
        exdir_channel_group.attrs['start_time'] = 0 * pq.s
        exdir_channel_group.attrs['stop_time'] = intan_file.duration
        exdir_channel_group.attrs["electrode_identities"] = channel_identities
        exdir_channel_group.attrs["electrode_idx"] = channel_identities - channel_identities[0]
        exdir_channel_group.attrs['electrode_group_id'] = intan_channel_group.channel_group_id
        # TODO else: test if attrs are the same
    return exdir_channel_groups


def generate_events(exdir_path, intan_rec):
    exdir_file = exdir.File(exdir_path, plugins=exdir.plugins.quantities)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)
    events = epochs.require_group('intan-epochs')

    for event_source in intan_rec.digital_in_events:
        ev_group = events.require_group('digital_in_' + str(int(np.unique(event_source.channels))))
        timestamps, durations, data = _get_epochs_from_event(event_source)
        times_dset = ev_group.require_dataset('timestamps', data=timestamps)
        times_dset.attrs['num_samples'] = len(timestamps)
        dur_dset = ev_group.require_dataset("durations", data=durations)
        dur_dset.attrs['num_samples'] = len(durations)
        dset = ev_group.require_dataset("data", data=data)
        dset.attrs['num_samples'] = len(data)

    for event_source in intan_rec.digital_out_events:
        ev_group = events.require_group('digital_out_' + str(int(np.unique(event_source.channels))))
        timestamps, durations, data = _get_epochs_from_event(event_source)
        times_dset = ev_group.require_dataset('timestamps', data=timestamps)
        times_dset.attrs['num_samples'] = len(timestamps)
        dur_dset = ev_group.require_dataset("durations", data=durations)
        dur_dset.attrs['num_samples'] = len(durations)
        dset = ev_group.require_dataset("data", data=data)
        dset.attrs['num_samples'] = len(data)


def _get_epochs_from_event(event):
    state_on_idxs = np.where(event.channel_states == 1)
    state_off_idxs = np.where(event.channel_states == -1)
    timestamps = event.times[state_on_idxs]

    if len(state_off_idxs[0]) == 0:
        durations = np.zeros(len(timestamps))
        data = event.channels
    else:
        if len(state_on_idxs[0]) == len(state_off_idxs[0]):
            durations = event.times[state_off_idxs] - event.times[state_on_idxs]
            data = event.channels[state_on_idxs]
        else:
            timestamps, durations, data = [], [], []
            unit = event.times.units
            for i, (st, st_1) in enumerate(zip(event.channel_states[:-1], event.channel_states[1:])):
                if i < len(event.channel_states) - 2:
                    if st == 1:
                        if st_1 == -1:
                            durations.append(event.times[i+1] - event.times[i])
                            timestamps.append(event.times[i])
                            data.append(event.channels[i])
                        else:
                            durations.append(0)
                            timestamps.append(event.times[i])
                            data.append(event.channels[i])
                else:
                    if st == 1:
                        durations.append(0)
                        timestamps.append(event.times[i])
                        data.append(event.channels[i])

            timestamps = np.array(timestamps) * unit
            durations = np.array(durations) * unit
            data = np.array(data)

    return timestamps, durations, data