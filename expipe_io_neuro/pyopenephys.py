"""
Python library for reading OpenEphys files.
Depends on: sys
            os
            glob
            datetime
            numpy
            quantities

Authors: Alessio Buccino @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Milad H. Mobarhan @CINPLA,
         Mikkel E. Lepperod @CINPLA
"""

# TODO: add extensive funciton descrption and verbose option for prints

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import sys
import quantities as pq
import os
import os.path as op
import glob
import numpy as np
import xml.etree.ElementTree as ET
from xmljson import yahoo as yh
from datetime import datetime
from six import exec_
from copy import copy
import locale
import struct


# TODO related files
# TODO append .continuous files directly to file and memory map in the end
# TODO ChannelGroup class - needs probe file
# TODO Channel class


# TODO add SYNC and TRACKERSTIM metadata

MAX_NUMBER_OF_EVENTS = int(1e6)

def _read_python(path):
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


class Channel:
    def __init__(self, index, name, gain, channel_id):
        self.index = index
        self.id = channel_id
        self.name = name
        self.gain = gain


class ChannelGroup:
    def __init__(self, channel_group_id, filename, channels, attrs):
        self.attrs = attrs
        self.filename = filename
        self.channel_group_id = channel_group_id
        self.channels = channels

    def __str__(self):
        return "<OpenEphys channel_group {}: channel_count: {}>".format(
            self.channel_group_id, len(self.channels)
        )


class AnalogSignal:
    def __init__(self, channel_id, signal, sample_rate):
        self.signal = signal
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    @property
    def times(self):
        if self.signal.shape[0] > 0:
            nsamples = self.signal.shape[1]
            return np.arange(nsamples) / self.sample_rate
        else:
            return np.array([])

    def __str__(self):
        return "<OpenEphys analog signal:shape: {}, sample_rate: {}>".format(
            self.signal.shape, self.sample_rate
        )


class DigitalSignal:
    def __init__(self, times, channel_id, sample_rate):
        self.times = times
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<OpenEphys digital signal: nchannels: {}>".format(
            self.channel_id
        )


class Sync:
    def __init__(self, times, channel_id, sample_rate):
        self.times = times
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<OpenEphys sync signal: nchannels: {}>".format(
            self.channel_id
        )


class TrackingData:
    def __init__(self, times, positions, attrs):
        self.attrs = attrs
        self.times = times
        self.positions = positions

    def __str__(self):
        return "<OpenEphys tracking data: times shape: {}, positions shape: {}>".format(
            self.times.shape, self.positions.shape
        )


class File:
    """
    Class for reading experimental data from an OpenEphys dataset.
    """
    def __init__(self, foldername, probefile=None, keep_channels=None, zero_channel=None):
        # TODO assert probefile is a probefile
        # TODO add default prb map and allow to add it later
        self._absolute_foldername = foldername
        self._path, relative_foldername = os.path.split(foldername)
        self._analog_signals_dirty = True
        self._digital_signals_dirty = True
        self._channel_groups_dirty = True
        self._tracking_dirty = True
        self._events_dirty = True

        self._keep_channels = keep_channels

        # TODO: support for multiple exp in same folder
        filenames = [f for f in os.listdir(self._absolute_foldername)]
        if not any(sett == 'settings.xml' for sett in filenames):
            raise ValueError("'setting.xml' should be in the folder")

        if not any('.eventsmessages' in f for f in filenames):
            raise ValueError("'.eventsmessages' should be in the folder")
        else:
            messagefile = [f for f in filenames if '.eventsmessages' in f][0]
            with open(op.join(self._absolute_foldername, messagefile), "r") as fh:
                self._software_sample_rate, self._start_exp = self._read_software_rate(fh)

        self.rhythm = False
        self.rhythmID = []
        rhythmRates = np.array([1., 1.25, 1.5, 2, 2.5, 3, 3.33, 4., 5., 6.25,
                                8., 10., 12.5, 15., 20., 25., 30.])
        self.osc = False
        self.oscID = []
        self.oscPort = []
        self.oscAddress = []
        self.tracking_timesamples_rate = 1000 * 1000. * pq.Hz

        self.sync = False
        self.syncID = []

        print('Loading Open-Ephys: reading settings.xml...')
        with open(op.join(self._absolute_foldername, 'settings.xml')) as f:
            xmldata = f.read()
            self.settings = yh.data(ET.fromstring(xmldata))['SETTINGS']
        # read date in US formate
        locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
        self._start_datetime = datetime.strptime(self.settings['INFO']['DATE'], '%d %b %Y %H:%M:%S')
        self._channel_info = {}
        self.nchan = 0
        FPGA_count = 0
        if type(self.settings['SIGNALCHAIN']) is list:
            for sigchain in self.settings['SIGNALCHAIN']:
                if type(sigchain['PROCESSOR']) is list:
                    for processor in sigchain['PROCESSOR']:
                        # print(processor['name'])
                        if processor['name'] == 'Sources/Rhythm FPGA':
                            assert FPGA_count == 0
                            FPGA_count += 1
                            # TODO can there be multiple FPGAs ?
                            self._channel_info['channels'] = []
                            self._channel_info['gain'] = []
                            self.rhythm = True
                            self.rhythmID = processor['NodeId']
                            gain = {ch['number']: ch['gain']
                                    for chs in processor['CHANNEL_INFO'].values()
                                    for ch in chs}
                            for chan in processor['CHANNEL']:
                                if chan['SELECTIONSTATE']['record'] == '1':
                                    self.nchan += 1
                                    chnum = chan['number']
                                    self._channel_info['channels'].append(int(chnum))
                                    self._channel_info['gain'].append(float(gain[chnum]))
                                sampleIdx = int(processor['EDITOR']['SampleRate'])-1
                                self._sample_rate = rhythmRates[sampleIdx] * 1000. * pq.Hz
                        if processor['name'] == 'Sources/OSC Port':
                            self.osc = True
                            self.oscID.append(processor['NodeId'])
                            self.oscPort.append(processor['EDITOR']['OSCNODE']['port'])
                            self.oscAddress.append(processor['EDITOR']['OSCNODE']['address'])
                        if processor['name'] == 'Sources/Sync Port':
                            self.sync = True
                            self.syncID = processor['NodeId']
                else:
                    processor = sigchain['PROCESSOR']
                    # print(processor['name'])
                    if processor['name'] == 'Sources/Rhythm FPGA':
                        assert FPGA_count == 0
                        FPGA_count += 1
                        # TODO can there be multiple FPGAs ?
                        self._channel_info['channels'] = []
                        self._channel_info['gain'] = []
                        self.rhythm = True
                        self.rhythmID = processor['NodeId']
                        gain = {ch['number']: ch['gain']
                                for chs in processor['CHANNEL_INFO'].values()
                                for ch in chs}
                        for chan in processor['CHANNEL']:
                            if chan['SELECTIONSTATE']['record'] == '1':
                                self.nchan += 1
                                chnum = chan['number']
                                self._channel_info['channels'].append(int(chnum))
                                self._channel_info['gain'].append(float(gain[chnum]))
                            sampleIdx = int(processor['EDITOR']['SampleRate']) - 1
                            self._sample_rate = rhythmRates[sampleIdx] * 1000. * pq.Hz
                        print('RhythmFPGA with ', self.nchan, ' channels. NodeId: ', self.rhythmID)
                    if processor['name'] == 'Sources/OSC Port':
                        self.osc = True
                        self.oscID.append(processor['NodeId'])
                        self.oscPort.append(processor['EDITOR']['OSCNODE']['port'])
                        self.oscAddress.append(processor['EDITOR']['OSCNODE']['address'])
                    if processor['name'] == 'Sources/Sync Port':
                        self.sync = True
                        self.syncID = processor['NodeId']
        else:
            sigchain = self.settings['SIGNALCHAIN']
            if type(sigchain['PROCESSOR']) is list:
                for processor in sigchain['PROCESSOR']:
                    # print(processor['name'])
                    if processor['name'] == 'Sources/Rhythm FPGA':
                        assert FPGA_count == 0
                        FPGA_count += 1
                        # TODO can there be multiple FPGAs ?
                        self._channel_info['channels'] = []
                        self._channel_info['gain'] = []
                        self.rhythm = True
                        self.rhythmID = processor['NodeId']
                        gain = {ch['number']: ch['gain']
                                for chs in processor['CHANNEL_INFO'].values()
                                for ch in chs}
                        for chan in processor['CHANNEL']:
                            if chan['SELECTIONSTATE']['record'] == '1':
                                self.nchan += 1
                                chnum = chan['number']
                                self._channel_info['channels'].append(int(chnum))
                                self._channel_info['gain'].append(float(gain[chnum]))
                            sampleIdx = int(processor['EDITOR']['SampleRate']) - 1
                            self._sample_rate = rhythmRates[sampleIdx] * 1000. * pq.Hz
                    if processor['name'] == 'Sources/OSC Port':
                        self.osc = True
                        self.oscID.append(processor['NodeId'])
                        self.oscPort.append(processor['EDITOR']['OSCNODE']['port'])
                        self.oscAddress.append(processor['EDITOR']['OSCNODE']['address'])
                    if processor['name'] == 'Sources/Sync Port':
                        self.sync = True
                        self.syncID = processor['NodeId']
            else:
                processor = sigchain['PROCESSOR']
                # print(processor['name'])
                if processor['name'] == 'Sources/Rhythm FPGA':
                    assert FPGA_count == 0
                    FPGA_count += 1
                    # TODO can there be multiple FPGAs ?
                    self._channel_info['channels'] = []
                    self._channel_info['gain'] = []
                    self.rhythm = True
                    self.rhythmID = processor['NodeId']
                    gain = {ch['number']: ch['gain']
                            for chs in processor['CHANNEL_INFO'].values()
                            for ch in chs}
                    for chan in processor['CHANNEL']:
                        if chan['SELECTIONSTATE']['record'] == '1':
                            self.nchan += 1
                            chnum = chan['number']
                            self._channel_info['channels'].append(int(chnum))
                            self._channel_info['gain'].append(float(gain[chnum]))
                        sampleIdx = int(processor['EDITOR']['SampleRate']) - 1
                        self._sample_rate = rhythmRates[sampleIdx] * 1000. * pq.Hz
                    print('RhythmFPGA with ', self.nchan, ' channels. NodeId: ', self.rhythmID)
                if processor['name'] == 'Sources/OSC Port':
                    self.osc = True
                    self.oscID.append(processor['NodeId'])
                    self.oscPort.append(processor['EDITOR']['OSCNODE']['port'])
                    self.oscAddress.append(processor['EDITOR']['OSCNODE']['address'])
                if processor['name'] == 'Sources/Sync Port':
                    self.sync = True
                    self.syncID = processor['NodeId']

        # Check openephys format
        if self.settings['CONTROLPANEL']['recordEngine'] == 'OPENEPHYS':
            self._format = 'openephys'
        elif self.settings['CONTROLPANEL']['recordEngine'] == 'RAWBINARY':
            self._format = 'binary'
        else:
            self._format = None
        print('Decoding data from ', self._format, ' format')

        if self.rhythm:
            print('RhythmFPGA with ', self.nchan, ' channels. NodeId: ', self.rhythmID)
        if self.osc:
            print('OSC Port. NodeId: ', self.oscID)

        if self.rhythm:
            sort_idx = np.argsort(self._channel_info['channels'])
            self._channel_info['channels'] = np.array(self._channel_info['channels'])[sort_idx]
            self._channel_info['gain'] = np.array(self._channel_info['gain'])[sort_idx]
            if probefile is not None:
                self._channel_group_info = _read_python(probefile)['channel_groups']
                for group in self._channel_group_info.values():
                    group['filemap'] = []
                    group['gain'] = []
                    # prb file channels are sequential, 'channels' are not as they depend on FPGA channel selection -> Collapse them into array
                    for chan in group['channels']:
                        idx = self._channel_info['channels'].tolist()[chan]
                        group['filemap'].append(idx)
                        group['gain'].append(self._channel_info['gain'][chan])
            else:
                # TODO sequential channel mapping
                print('sequential channel mapping')

    @property
    def session(self):
        return op.split(self._absolute_foldername)[-1]

    @property
    def duration(self):
        if self.rhythm:
            self._duration = (self.analog_signals[0].signal.shape[1] /
                              self.analog_signals[0].sample_rate)
        elif self.osc:
            self._duration = (len(self.tracking[0].positions[0]) /
                              self.tracking[0].attrs['sample_rate'])
        else:
            self._duration = []

        return self._duration

    @property
    def sample_rate(self):
        if self.rhythm:
            return self._sample_rate
        else:
            return self._software_sample_rate


    def channel_group(self, channel_id):
        if self._channel_groups_dirty:
            self._read_channel_groups()

        return self._channel_id_to_channel_group[channel_id]

    @property
    def channel_groups(self):
        if self._channel_groups_dirty:
            self._read_channel_groups()

        return self._channel_groups

    @property
    def analog_signals(self):
        if self._analog_signals_dirty:
            self._read_analog_signals()

        return self._analog_signals

    @property
    def digital_signals(self):
        if self._digital_signals_dirty:
            self._read_digital_signals()

        return self._digital_signals

    @property
    def sync_signals(self):
        if self._digital_signals_dirty:
            self._read_digital_signals()

        return self._sync_signals

    @property
    def events(self):
        if self._events_dirty:
            self._read_digital_signals()

        return self._events

    @property
    def tracking(self):
        if self._tracking_dirty:
            self._read_tracking()

        return self._tracking


    def _read_software_rate(self, fh):
        spl = fh.readline().split()
        if any(['Software' in s for s in spl]):
            stime = spl[-1]
            stime = stime.split('@')
            start = stime[0]
            sample_rate = float(stime[-1][:-2]) * pq.Hz
        else:
            start = sample_rate = []

        return sample_rate, start


    def _read_channel_groups(self):
        self._channel_id_to_channel_group = {}
        self._channel_group_id_to_channel_group = {}
        self._channel_count = 0
        self._channel_groups = []
        for channel_group_id, channel_group_content in self._channel_group_info.items():
            num_chans = len(channel_group_content['channels'])
            self._channel_count += num_chans
            channels = []
            for idx, channel_id in enumerate(channel_group_content['filemap']):
                channel = Channel(
                    index=idx,
                    channel_id=channel_id,
                    name="channel_{}_channel_group_{}".format(channel_id,
                                                              channel_group_id),
                    gain=channel_group_content['gain'][idx]
                )
                channels.append(channel)

            channel_group = ChannelGroup(
                channel_group_id=channel_group_id,
                filename=None,#TODO,
                channels=channels,
                attrs=None #TODO
            )
            ana = self.analog_signals[0]
            analog_signals = []
            for channel in channels:
                analog_signals.append(AnalogSignal(signal=ana.signal[channel.id],
                                                   channel_id=channel.id,
                                                   sample_rate=ana.sample_rate))

            channel_group.analog_signals = analog_signals

            self._channel_groups.append(channel_group)
            self._channel_group_id_to_channel_group[channel_group_id] = channel_group

            for channel_id in channel_group_content['channels']:
                self._channel_id_to_channel_group[channel_id] = channel_group

        # TODO channel mapping to file
        self._channel_ids = np.arange(self._channel_count)
        self._channel_groups_dirty = False

    def _read_tracking(self):
        filenames = [f for f in os.listdir(self._absolute_foldername)]
        if self.osc is True and any('.eventsbinary' in f for f in filenames):
            posfile = [f for f in filenames if '.eventsbinary' in f][0]
            print('.eventsbinary: ', posfile)
            with open(op.join(self._absolute_foldername, posfile), "rb") as fh: #, encoding='utf-8', errors='ignore') as fh:
                self._read_tracking_events(fh)
        else:
            raise ValueError("'.eventsbinary' should be in the folder")

    def _read_tracking_events(self, fh):
        print('Reading positions...')

        # TODO consider NOT writing header from openephys
        header = readHeader(fh)

        if float(header['version']) < 0.4:
            raise Exception('Loader is only compatible with .events files with version 0.4 or higher')

        # TODO consider reading the entire file and unpack it
        struct_fmt = '=Bq4f'  # int[5], float, byte[255]
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from

        nsamples = (os.fstat(fh.fileno()).st_size -fh.tell()) // struct_len
        print('Estimated position samples: ', nsamples)
        nread = 0

        read_data=[]
        while True:
            bytes = fh.read(struct_len)
            if not bytes:
                break
            s = struct_unpack(bytes)
            read_data.append(s)
            nread+=1

        print('Read position samples: ', nread)

        ids, timestamps, x, y, w, h = zip(*read_data)
        ids = np.array(ids)
        timestamps = np.array(timestamps)
        x = np.array(x)
        y = np.array(y)
        w = np.array(w)
        h = np.array(h)

        ts = timestamps / 1000.

        # Sort out different Sources
        if len(np.unique(ids)) == 1:
            print("Single tracking source")

            # adjust times with linear interpolation
            # idx_non_zero = np.where(ts != 0)
            # linear_coeff = np.polyfit(np.arange(len(ts))[idx_non_zero], ts[idx_non_zero], 1)
            # times_fit = linear_coeff[0]*(np.arange(len(ts))) + linear_coeff[1]
            difft = np.diff(ts)
            avg_period = np.mean(difft)
            sample_rate_s = 1./float(avg_period) * pq.Hz

            # Camera (0,0) is top left corner -> adjust y
            coord_s = np.array([x, 1-y])
            ts_s = ts

            width_s = np.mean(w)
            height_s = np.mean(h)

            attrs = dict()
            attrs['sample_rate'] = sample_rate_s
            attrs['length_scale'] = np.array([width_s, height_s])

        else:
            print("Multiple tracking sources")
            sources = np.unique(ids)
            coord_s, w_s, h_s, ts_s = [], [], [], []
            sample_rate_s, width_s, height_s = [], [], []
            for ss in sources:
                x_ = np.squeeze(x[np.where(ids==ss)])
                y_ = np.squeeze(y[np.where(ids==ss)])
                w_ = np.squeeze(w[np.where(ids==ss)])
                h_ = np.squeeze(h[np.where(ids==ss)])
                ts_ = np.squeeze(ts[np.where(ids==ss)])

                # adjust times with linear interpolation
                # idx_non_zero = np.where(ts_ != 0)
                # linear_coeff = np.polyfit(np.arange(len(ts_))[idx_non_zero], ts_[idx_non_zero], 1)
                # times_fit = linear_coeff[0]*(np.arange(len(ts_))) + linear_coeff[1]
                difft = np.diff(ts_)
                avg_period = np.mean(difft)
                sample_rate_ = 1./float(avg_period) * pq.Hz
                # times_fit = ts_
                # sample_rate_ = 0

                # Camera (0,0) is top left corner -> adjust y
                coord_ = np.array([x_, 1-y_])
                coord_s.append(coord_)
                ts_s.append(ts_)

                sample_rate_s.append(sample_rate_)
                width_s.append(np.mean(w_))
                height_s.append(np.mean(h_))

            attrs = dict()
            attrs['sample_rate'] = np.array(sample_rate_s)
            attrs['length_scale'] = np.array([width_s, height_s])
            attrs['nodeId'] = self.oscID
            attrs['port'] = self.oscPort
            attrs['address'] = self.oscAddress

        tracking_data = [TrackingData(
            times=ts_s,
            positions=coord_s,
            attrs=attrs
        )]

        self._tracking = tracking_data
        # self._tracking=results
        self._tracking_dirty = False

    def _read_analog_signals(self):
        if self.rhythm:
            # Check and decode files
            filenames = [f for f in os.listdir(self._absolute_foldername)]
            anas = np.array([])
            timestamps = np.array([])
            if self._format == 'binary':
                if self.rhythm is True:
                    if any('.dat' in f for f in filenames):
                        datfile = [f for f in filenames if '.dat' in f and 'experiment' in f][0]
                        print('.dat: ', datfile)
                        with open(op.join(self._absolute_foldername, datfile), "rb") as fh:
                            anas, nsamples = read_analog_binary_signals(fh, self.nchan)
                        # Keep only selected channels
                        if self._keep_channels is not None:
                            # Compare recorded and keep_channels to find idx
                            idx = []
                            for ch in self._keep_channels:
                                if len(np.argwhere(self._channel_info['channels'] == ch)) == 1:
                                    idx.append(np.argwhere(self._channel_info['channels'] == ch).item())
                                elif len(np.argwhere(self._channel_info['channels'] == ch)) == 0:
                                    raise ValueError('Some channels in keep_channels were not recorded')
                            anas_keep = anas[idx].astype('float32')
                            for i, ch in enumerate(idx):
                                anas_keep[i] = anas_keep[i]*self._channel_info['gain'][ch]
                        else:
                            anas_keep = anas.astype('float32')
                            for ch, i in self._channel_info['channels']:
                                anas_keep[i] = float(anas_keep[i])*self._channel_info['gain'][ch]
                    else:
                        raise ValueError("'experiment_###.dat' should be in the folder")
                else:
                    print('No rhythm FPGA data')
            elif self._format == 'openephys':
                # Find continuous CH data
                contFiles = [f for f in os.listdir(self._absolute_foldername) if 'continuous' in f and 'CH' in f]
                contFiles = sorted(contFiles)
                if len(contFiles) != 0:
                    print('Reading all channels')
                    for f in contFiles:
                        fullpath = op.join(self._absolute_foldername, f)
                        sig = read_analog_continuous_signal(fullpath)
                        if anas.shape[0] < 1:
                            anas = sig['data'][None, :]
                        else:
                            if sig['data'].size == anas[-1].size:
                                anas = np.append(anas, sig['data'][None, :], axis=0)
                            else:
                                raise Exception('Channels must have the same number of samples')

                    anas = np.array(anas)
                    print('Done!')

            self._analog_signals = [AnalogSignal(
                channel_id=range(anas_keep.shape[0]),
                signal=anas_keep,
                sample_rate=self.sample_rate
            )]
        else:
            self._analog_signals = [AnalogSignal(
                channel_id=np.array([]),
                signal=np.array([]),
                sample_rate=self.sample_rate
            )]

        self._analog_signals_dirty = False

    def _read_digital_signals(self):
        filenames = [f for f in os.listdir(self._absolute_foldername)]
        if any('.events' in f and 'all_channels' in f for f in filenames):
            eventsfile = [f for f in filenames if '.events' in f and 'all_channels' in f][0]
            print('.events ', eventsfile)
            with open(op.join(self._absolute_foldername, eventsfile), "rb") as fh: #, encoding='utf-8', errors='ignore') as fh:
                data = {}

                print('loading events...')
                header = readHeader(fh)

                if float(header['version']) < 0.4:
                    raise Exception('Loader is only compatible with .events files with version 0.4 or higher')

                data['header'] = header

                struct_fmt = '=qH4BH'  # int[5], float, byte[255]
                struct_len = struct.calcsize(struct_fmt)
                struct_unpack = struct.Struct(struct_fmt).unpack_from

                nsamples = (os.fstat(fh.fileno()).st_size - fh.tell()) // struct_len
                print('Estimated events samples: ', nsamples)
                nread = 0

                read_data = []
                while True:
                    bytes = fh.read(struct_len)
                    if not bytes:
                        break
                    s = struct_unpack(bytes)
                    read_data.append(s)
                    nread += 1

                print('Read event samples: ', nread)

                timestamps, sampleNum, eventType, nodeId, eventId, channel, recordingNumber = zip(*read_data)

                timestamps = np.array(timestamps)
                sampleNum = np.array(sampleNum)
                nodeId = np.array(nodeId)
                eventType = np.array(eventType)
                eventId = np.array(eventId)
                channel = np.array(channel)
                recordingNumber = np.array(recordingNumber)

                # index = -1
                #
                # channel = np.zeros(MAX_NUMBER_OF_EVENTS)
                # timestamps = np.zeros(MAX_NUMBER_OF_EVENTS)
                # sampleNum = np.zeros(MAX_NUMBER_OF_EVENTS)
                # nodeId = np.zeros(MAX_NUMBER_OF_EVENTS)
                # eventType = np.zeros(MAX_NUMBER_OF_EVENTS)
                # eventId = np.zeros(MAX_NUMBER_OF_EVENTS)
                # recordingNumber = np.zeros(MAX_NUMBER_OF_EVENTS)
                #
                # while fh.tell() < os.fstat(fh.fileno()).st_size:
                #
                #     index += 1
                #
                #     timestamps[index] = np.fromfile(fh, np.dtype('<i8'), 1)
                #     sampleNum[index] = np.fromfile(fh, np.dtype('<i2'), 1)
                #     eventType[index] = np.fromfile(fh, np.dtype('<u1'), 1)
                #     nodeId[index] = np.fromfile(fh, np.dtype('<u1'), 1)
                #     eventId[index] = np.fromfile(fh, np.dtype('<u1'), 1)
                #     channel[index] = np.fromfile(fh, np.dtype('<u1'), 1)
                #     recordingNumber[index] = np.fromfile(fh, np.dtype('<u2'), 1)

                data['channel'] = channel
                data['timestamps'] = timestamps
                data['eventType'] = eventType
                data['nodeId'] = nodeId
                data['eventId'] = eventId
                data['recordingNumber'] = recordingNumber
                data['sampleNum'] = sampleNum

                # TODO: check if data is null (data['event...'] is null?
                # Consider only TTL from FPGA (for now)
                if self.rhythm:
                    if len(data['timestamps']) != 0:
                        idxttl_fpga = np.where((data['eventType'] == 3) & (data['nodeId'] == int(self.rhythmID)))
                        digchan = []
                        digs = []
                        if len(idxttl_fpga[0]) != 0:
                            print('TTLevents: ', len(idxttl_fpga[0]))
                            digchan = np.unique(data['channel'][idxttl_fpga])
                            if len(digchan) == 1:
                                # Single digital input
                                digs = data['timestamps'][idxttl_fpga]
                                # Consider rising edge only
                                digs = digs[::2]
                                # remove start_time (offset) and transform in seconds
                                digs -= data['timestamps'][0]
                                digs = digs.astype(dtype='float')/self.sample_rate
                            else:
                                for chan in digchan:
                                    idx_chan = np.where(data['channel'] == chan)
                                    new_dig = data['timestamps'][idx_chan]
                                    # Consider rising edge only
                                    new_dig = new_dig[::2]
                                    new_dig -= data['timestamps'][0]
                                    new_dig = new_dig.astype(dtype='float')/self.sample_rate
                                    digs.append(new_dig)

                        self._digital_signals = [DigitalSignal(
                            channel_id=digchan,
                            times=digs,
                            sample_rate=self.sample_rate
                        )]
                    else:
                        self._digital_signals = [DigitalSignal(
                            channel_id=np.array([]),
                            times=np.array([]),
                            sample_rate=[]
                        )]
                else:
                    self._digital_signals = [DigitalSignal(
                        channel_id=np.array([]),
                        times=np.array([]),
                        sample_rate=[]
                    )]

                if self.sync:
                    if len(data['timestamps']) != 0:
                        idxttl_sync = np.where((data['eventType'] == 3) & (data['nodeId'] == int(self.syncID)))
                        syncchan = []
                        syncs = []
                        if len(idxttl_sync[0]) != 0:
                            print('TTL Sync events: ', len(idxttl_sync[0]))
                            syncchan = np.unique(data['channel'][idxttl_sync])
                            if len(syncchan) == 1:
                                # Single digital input
                                syncs = data['timestamps'][idxttl_sync]
                                # remove start_time (offset) and transform in seconds
                                syncs -= data['timestamps'][0]
                                syncs /= self.sample_rate
                            else:
                                for chan in syncchan:
                                    idx_chan = np.where(data['channel'] == chan)
                                    new_sync = data['timestamps'][idx_chan]

                                    new_sync -= data['timestamps'][0]
                                    new_sync /= self.sample_rate
                                    syncs.append(new_sync)

                        self._sync_signals = [Sync(
                            channel_id=syncchan,
                            times=syncs,
                            sample_rate=self.sample_rate
                        )]
                    else:
                        self._sync_signals = [DigitalSignal(
                            channel_id=np.array([]),
                            times=np.array([]),
                            sample_rate=[]
                        )]
                else:
                    self._sync_signals = [DigitalSignal(
                        channel_id=np.array([]),
                        times=np.array([]),
                        sample_rate=[]
                    )]

                self._digital_signals_dirty = False
                self._events_dirty = False
                self._events = data


def read_analog_binary_signals(filehandle, numchan):

    numchan=int(numchan)

    nsamples = os.fstat(filehandle.fileno()).st_size // (numchan*2)
    print('Estimated samples: ', int(nsamples), ' Numchan: ', numchan)

    samples = np.memmap(filehandle, np.dtype('i2'), mode='r',
                        shape=(nsamples, numchan))
    samples = np.transpose(samples)

    return samples, nsamples


def read_analog_continuous_signal(filepath, dtype=float, verbose=False,
    start_record=None, stop_record=None, ignore_last_record=True):
    """Load continuous data from a single channel in the file `filepath`.

    This is intended to be mostly compatible with the previous version.
    The differences are:
    - Ability to specify start and stop records
    - Converts numeric data in the header from string to numeric data types
    - Does not rely on a predefined maximum data size
    - Does not necessarily drop the last record, which is usually incomplete
    - Uses the block length that is specified in the header, instead of
        hardcoding it.
    - Returns timestamps and recordNumbers as int instead of float
    - Tests the record metadata (N and record marker) for internal consistency

    The OpenEphys file format breaks the data stream into "records",
    typically of length 1024 samples. There is only one timestamp per record.

    Args:
        filepath : string, path to file to load
        dtype : float or np.int16
            If float, then the data will be multiplied by bitVolts to convert
            to microvolts. This increases the memory required by 4 times.
        verbose : whether to print debugging messages
        start_record, stop_record : indices that control how much data
            is read and returned. Pythonic indexing is used,
            so `stop_record` is not inclusive. If `start` is None, reading
            begins at the beginning; if `stop` is None, reading continues
            until the end.
        ignore_last_record : The last record in the file is almost always
            incomplete (padded with zeros). By default it is ignored, for
            compatibility with the old version of this function.

    Returns: dict, with following keys
        data : array of samples of data
        header : the header info, as returned by readHeader
        timestamps : the timestamps of each record of data that was read
        recordingNumber : the recording number of each record of data that
            was read. The length is the same as `timestamps`.
    """
    if dtype not in [float, np.int16]:
        raise ValueError("Invalid data type. Must be float or np.int16")

    if verbose:
        print("Loading continuous data from " + filepath)

    """Here is the OpenEphys file format:
    'each record contains one 64-bit timestamp, one 16-bit sample
    count (N), 1 uint16 recordingNumber, N 16-bit samples, and
    one 10-byte record marker (0 1 2 3 4 5 6 7 8 255)'
    Thus each record has size 2*N + 22 bytes.
    """
    # This is what the record marker should look like
    spec_record_marker = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

    # Lists for data that's read
    timestamps = []
    recordingNumbers = []
    samples = []
    samples_read = 0
    records_read = 0

    # Open the file
    with file(filepath, 'rb') as f:
        # Read header info, file length, and number of records
        header = readHeader(f)
        record_length_bytes = 2 * header['blockLength'] + 22
        fileLength = os.fstat(f.fileno()).st_size
        n_records = get_number_of_records(filepath)

        # Use this to set start and stop records if not specified
        if start_record is None:
            start_record = 0
        if stop_record is None:
            stop_record = n_records

        # We'll stop reading after this many records are read
        n_records_to_read = stop_record - start_record

        # Seek to the start location, relative to the current position
        # right after the header.
        f.seek(record_length_bytes * start_record, 1)

        # Keep reading till the file is finished
        while f.tell() < fileLength and records_read < n_records_to_read:
            # Skip the last record if requested, which usually contains
            # incomplete data
            if ignore_last_record and f.tell() == (
                fileLength - record_length_bytes):
                break

            # Read the timestamp for this record
            # litte-endian 64-bit signed integer
            timestamps.append(np.fromfile(f, np.dtype('<i8'), 1))

            # Read the number of samples in this record
            # little-endian 16-bit unsigned integer
            N = np.fromfile(f, np.dtype('<u2'), 1).item()
            if N != header['blockLength']:
                raise IOError('Found corrupted record in block')

            # Read and store the recording numbers
            # big-endian 16-bit unsigned integer
            recordingNumbers.append(np.fromfile(f, np.dtype('>u2'), 1))

            # Read the data
            # big-endian 16-bit signed integer
            data = np.fromfile(f, np.dtype('>i2'), N)
            if len(data) != N:
                raise IOError("could not load the right number of samples")

            # Optionally convert dtype
            if dtype == float:
                data = data * header['bitVolts']

            # Store the data
            samples.append(data)

            # Extract and test the record marker
            record_marker = np.fromfile(f, np.dtype('<u1'), 10)
            if np.any(record_marker != spec_record_marker):
                raise IOError("corrupted record marker at record %d" %
                    records_read)

            # Update the count
            samples_read += len(samples)
            records_read += 1

    # Concatenate results, or empty arrays if no data read (which happens
    # if start_sample is after the end of the data stream)
    res = {'header': header}
    if samples_read > 0:
        res['timestamps'] = np.concatenate(timestamps)
        res['data'] = np.concatenate(samples)
        res['recordingNumber'] = np.concatenate(recordingNumbers)
    else:
        res['timestamps'] = np.array([], dtype=np.int)
        res['data'] = np.array([], dtype=dtype)
        res['recordingNumber'] = np.array([], dtype=np.int)

    return res


'''from OpenEphys.py'''
def readHeader(fh):
    """Read header information from the first 1024 bytes of an OpenEphys file.

    Args:
        f: An open file handle to an OpenEphys file

    Returns: dict with the following keys.
        - bitVolts : float, scaling factor, microvolts per bit
        - blockLength : int, e.g. 1024, length of each record (see
            loadContinuous)
        - bufferSize : int, e.g. 1024
        - channel : the channel, eg "'CH1'"
        - channelType : eg "'Continuous'"
        - date_created : eg "'15-Jun-2016 21212'" (What are these numbers?)
        - description : description of the file format
        - format : "'Open Ephys Data Format'"
        - header_bytes : int, e.g. 1024
        - sampleRate : float, e.g. 30000.
        - version: eg '0.4'
        Note that every value is a string, even numeric data like bitVolts.
        Some strings have extra, redundant single apostrophes.
    """
    header = {}

    # Read the data as a string
    # Remove newlines and redundant "header." prefixes
    # The result should be a series of "key = value" strings, separated
    # by semicolons.
    header_string = fh.read(1024).decode('utf-8').replace('\n','').replace('header.','')

    # Parse each key = value string separately
    for pair in header_string.split(';'):
        if '=' in pair:
            # print pair
            key, value = pair.split(' = ')
            key = key.strip()
            value = value.strip()

            # Convert some values to numeric
            if key in ['bitVolts', 'sampleRate']:
                header[key] = float(value)
            elif key in ['blockLength', 'bufferSize', 'header_bytes']:
                header[key] = int(value)
            else:
                # Keep as string
                header[key] = value

    return header


def get_number_of_records(filepath):
    # Open the file
    with file(filepath, 'rb') as f:
        # Read header info
        header = readHeader(f)

        # Get file length
        fileLength = os.fstat(f.fileno()).st_size

        # Determine the number of records
        record_length_bytes = 2 * header['blockLength'] + 22
        n_records = int((fileLength - 1024) / record_length_bytes)
        # if (n_records * record_length_bytes + 1024) != fileLength:
        #     print("file does not divide evenly into full records")
        #     # raise IOError("file does not divide evenly into full records")

    return n_records
