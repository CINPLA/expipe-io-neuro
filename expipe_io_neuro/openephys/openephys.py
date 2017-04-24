from . import pyopenephys
import exdir
import shutil
import glob
import os
import quantities as pq
import numpy as np

from expipe.io.core import Filerecord
from expipe.io.core import user
from expipe import settings
import os.path as op

# TODO inform database about openephys data being included
# TODO SpikeTrain class - needs klusta stuff


def _prepare_exdir_file(exdir_file):
    general = exdir_file.require_group("general")
    subject = general.require_group("subject")
    processing = exdir_file.require_group("processing")
    epochs = exdir_file.require_group("epochs")

    return general, subject, processing, epochs


def convert(openephys_file, exdir_path):
    exdir_file = exdir.File(exdir_path)
    dtime = openephys_file._start_datetime.strftime('%Y-%m-%dT%H:%M:%S')
    exdir_file.attrs['session_start_time'] = dtime
    exdir_file.attrs['session_duration'] = openephys_file.duration
    acquisition = exdir_file.require_group("acquisition")
    general = exdir_file.require_group("general")
    processing = exdir_file.require_group("processing")
    subject = general.require_group("subject")

    target_folder = op.join(acquisition.directory, openephys_file.session)
    acquisition.attrs["openephys_session"] = openephys_file.session
    acquisition.attrs["acquisition_system"] = 'OpenEphys'

    print("Copying ", openephys_file._absolute_foldername, " to ", target_folder)
    shutil.copytree(openephys_file._absolute_foldername, target_folder)


def _prepare_channel_groups(exdir_path, openephys_file):
    exdir_file = exdir.File(exdir_path)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)
    exdir_channel_groups = []
    elphys = processing.require_group('electrophysiology')
    for openephys_channel_group in openephys_file.channel_groups:
        exdir_channel_group = elphys.require_group(
            "channel_group_{}".format(openephys_channel_group.channel_group_id))
        exdir_channel_groups.append(exdir_channel_group)
        channel_identities = np.array([ch.index for ch in openephys_channel_group.channels])
        exdir_channel_group.attrs['start_time'] = 0 * pq.s
        exdir_channel_group.attrs['stop_time'] = openephys_file.duration
        exdir_channel_group.attrs["electrode_identities"] = channel_identities
        exdir_channel_group.attrs["electrode_idx"] = channel_identities - channel_identities[0]
        exdir_channel_group.attrs['electrode_group_id'] = openephys_channel_group.channel_group_id
        # TODO else: test if attrs are the same
    return exdir_channel_groups


def generate_lfp(exdir_path, openephys_file):
    import scipy.signal as ss
    import copy
    exdir_channel_groups = _prepare_channel_groups(exdir_path, openephys_file)
    for channel_group, openephys_channel_group in zip(exdir_channel_groups,
                                                      openephys_file.channel_groups):
        lfp = channel_group.require_group("LFP")
        group_id = openephys_channel_group.channel_group_id
        print('Generating LFP, channel group ', group_id)
        for channel in openephys_channel_group.channels:
                lfp_timeseries = lfp.require_group(
                    "LFP_timeseries_{}".format(channel.index)
                )
                analog_signal = openephys_channel_group.analog_signals[channel.index]
                # decimate
                target_rate = 1000 * pq.Hz
                signal = np.array(analog_signal.signal, dtype=float)
                sample_rate = copy.copy(analog_signal.sample_rate)
                qs = [10, int((analog_signal.sample_rate / target_rate) / 10)]
                for q in qs:
                    signal = ss.decimate(signal, q=q, zero_phase=True)
                    sample_rate /= q
                t_stop = len(signal) / sample_rate
                assert round(t_stop, 1) == round(openephys_file.duration, 1), '{}, {}'.format(t_stop, openephys_file.duration)
                signal = signal * channel.gain
                lfp_timeseries.attrs["num_samples"] = len(signal)
                lfp_timeseries.attrs["start_time"] = 0 * pq.s
                lfp_timeseries.attrs["stop_time"] = t_stop
                lfp_timeseries.attrs["sample_rate"] = sample_rate
                lfp_timeseries.attrs["electrode_identity"] = analog_signal.channel_id
                lfp_timeseries.attrs["electrode_idx"] = analog_signal.channel_id - openephys_channel_group.channel_group_id * 4
                lfp_timeseries.attrs['electrode_group_id'] = group_id
                data = lfp_timeseries.require_dataset("data", data=signal)
                data.attrs["num_samples"] = len(signal)
                # NOTE: In exdirio (python-neo) sample rate is required on dset #TODO
                data.attrs["sample_rate"] = sample_rate


def generate_spike_trains(exdir_path, openephys_file, source='klusta'):
    import neo
    if source == 'klusta':
        print('Generating spike trains from KWIK file')
        exdir_file = exdir.File(exdir_path)
        acquisition = exdir_file["acquisition"]
        openephys_session = acquisition.attrs["openephys_session"]
        openephys_directory = op.join(acquisition.directory, openephys_session)
        kwikfile = [f for f in os.listdir(openephys_directory) if f.endswith('_klusta.kwik')]
        if len(kwikfile) > 0:
            kwikfile = op.join(openephys_directory, kwikfile)[0]
            if op.exists(kwikfile):
                kwikio = neo.io.KwikIO(filename=kwikfile)
                blk = kwikio.read_block()
                exdirio = neo.io.ExdirIO(exdir_path)
                exdirio.write_block(blk)
        else:
            print('.kwik file is not in exdir folder')
    elif source == 'openephys':
        print(openephys_file.spiketains)
        # exdir_channel_groups = _prepare_channel_groups(exdir_path, openephys_file)
        # for channel_group, openephys_channel_group in zip(exdir_channel_groups,
        #                                                   openephys_file.channel_groups):
        #     group_id = openephys_channel_group.channel_group_id
        #     print('Loading spikes from, channel group ', group_id)
        #     print(channel_group.spiketains)
    else:
        raise ValueError(source + ' not supported')


def generate_tracking(exdir_path, openephys_file):
    exdir_file = exdir.File(exdir_path)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)
    tracking = processing.require_group('tracking')
    # NOTE openephys supports only one camera, but other setups might support several
    camera = tracking.require_group("camera_0")
    position = camera.require_group("Position")
    position.attrs['start_time'] = 0 * pq.s
    position.attrs['stop_time'] = openephys_file.duration
    tracking_data = openephys_file.tracking[0]
    for n, (times, coords) in enumerate(zip(tracking_data.times,
                                            tracking_data.positions)):
        led = position.require_group("led_" + str(n))
        dset = led.require_dataset('data', coords.transpose() * pq.m) # TODO units??
        dset.attrs['num_samples'] = coords.shape[1]
        dset = led.require_dataset("timestamps", times)
        dset.attrs['num_samples'] = len(times)
        led.attrs['start_time'] = 0 * pq.s
        led.attrs['stop_time'] = openephys_file.duration


class OpenEphysFilerecord(Filerecord):
    def __init__(self, action, filerecord_id=None):
        super().__init__(action, filerecord_id)

    def import_file(self, openephys_file):
        convert(openephys_file=openephys_file,
                exdir_path=self.local_path)

    def generate_tracking(self, openephys_file):
        generate_tracking(self.local_path, openephys_file)

    def generate_lfp(self, openephys_file):
        generate_analog_signals(self.local_path, openephys_file)

    def generate_spike_trains(self, openephys_file):
        generate_spike_trains(self.local_path, openephys_file)

    def generate_inp(self, openephys_file):
        generate_inp(self.local_path, openephys_file)


if __name__ == '__main__':
    openephys_directory = '/home/mikkel/Ephys/1715_2017-04-23_12-31-07_01'
    exdir_path = 'nada'
    probefile = '/home/mikkel/.config/expipe/tetrodes32ch-klusta-oe.prb'
    openephys_file = pyopenephys.File(openephys_directory, probefile)
    # if op.exists(exdir_path):
    #     shutil.rmtree(exdir_path)
    # convert(openephys_directory=openephys_file,
    #         exdir_path=exdir_path)
    # generate_tracking(exdir_path, openephys_file)
    # generate_lfp(exdir_path, openephys_file)
    generate_spike_trains(exdir_path, openephys_file, source='openephys')
    # generate_inp(exdir_path)
