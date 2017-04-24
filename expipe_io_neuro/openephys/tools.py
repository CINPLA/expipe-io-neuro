from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import quantities as pq
import os
import os.path as op
import numpy as np
from six import exec_
import locale
import struct
import platform


def _read_python(path):
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def _cut_to_same_len(*args):
    out = []
    lens = []
    for arg in args:
        lens.append(len(arg))
    minlen = min(lens)
    for arg in args:
        out.append(arg[:minlen])
    return tuple(out)


def _zeros_to_nan(*args):
    for arg in args:
        arg[arg == 0.0] = np.nan


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
        
        Taken from OpenEphys team
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
    with open(filepath, 'rb') as f:
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

# TODO require quantities and deal with it
def clip_anas(analog_signals, times, clipping_times, start_end):
    '''

    :param analog_signals:
    :param times:
    :param clipping_times:
    :param start_end:
    :return:
    '''

    if len(analog_signals.signal) != 0:
        times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        if len(analog_signals.signal.shape) == 2:
            anas_clip = analog_signals.signal[:, idx[0]]
        else:
            anas_clip = analog_signals.signal[idx[0]]

        return anas_clip
    else:
        return []


def clip_digs(digital_signals, clipping_times, start_end):
    '''

    :param digital_signals:
    :param clipping_times:
    :param start_end:
    :return:
    '''

    digs_clip = []
    for i, dig in enumerate(digital_signals.times):
        dig.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((dig > clipping_times[0]) & (dig < clipping_times[1]))
        elif len(clipping_times) == 1:
            if start_end == 'start':
                idx = np.where(dig > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(dig < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')
        digs_clip.append(dig[idx])

    return np.array(digs_clip) * pq.s


def clip_tracking(tracking, clipping_times, start_end):
    '''

    :param tracking:
    :param clipping_times:
    :param start_end:
    :return:
    '''
    assert len(tracking.positions) == len(tracking.times)

    track_clip = []
    t_clip = []

    for i, tr in enumerate(tracking.positions):
        tracking.times[i].rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((tracking.times[i] > clipping_times[0]) & (tracking.times[i] < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(tracking.times[i] > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(tracking.times[i] < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        track_clip.append(np.array([led[idx[0]] for led in tr]))
        times = tracking.times[i][idx[0]] - clipping_times[0]
        t_clip.append(times)

    return track_clip, t_clip


def clip_times(times, clipping_times, start_end):
    '''

    :param times:
    :param clipping_times:
    :param start_end:
    :return:
    '''
    times.rescale(pq.s)

    if len(clipping_times) == 2:
        idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
    elif len(clipping_times) ==  1:
        if start_end == 'start':
            idx = np.where(times > clipping_times[0])
        elif start_end == 'end':
            idx = np.where(times < clipping_times[0])
    else:
        raise AttributeError('clipping_times must be of length 1 or 2')
    times_clip = times[idx]

    return times_clip


def find_nearest(array, value, n=1, not_in_idx=None):

    if not_in_idx is None:
        if n==1:
            idx = (np.abs(array-value)).argmin()
        else:
            idx = (np.abs(array-value)).argsort()[:n]
        return array[idx], idx
    else:
        if len(array) != 0:
            left_idx = np.ones(len(array), dtype=bool)
            left_idx[not_in_idx] = False
            left_array=array[left_idx]
            if n==1:
                idx = (np.abs(left_array-value)).argmin()
            else:
                idx = (np.abs(left_array-value)).argsort()[:n]
            val = left_array[idx]
            idx = np.where(array==val)
            return array[idx], idx
        else:
            print('Array length must be greater than 0')
            return None, -1


def loadSpikes(filepath):
    
    # doesn't quite work...spikes are transposed in a weird way    
    
    data = { }
    
    print('loading spikes...')
    
    f = open(filepath,'rb')
    header = readHeader(f)
    
    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .spikes files with version 0.4 or higher')
     
    data['header'] = header 
    numChannels = int(header['num_channels'])
    numSamples = 40 # **NOT CURRENTLY WRITTEN TO HEADER**
    
    spikes = np.zeros((MAX_NUMBER_OF_SPIKES, numSamples, numChannels))
    timestamps = np.zeros(MAX_NUMBER_OF_SPIKES)
    source = np.zeros(MAX_NUMBER_OF_SPIKES)
    gain = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    thresh = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    sortedId = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    recNum = np.zeros(MAX_NUMBER_OF_SPIKES)
    
    currentSpike = 0
    
    while f.tell() < os.fstat(f.fileno()).st_size:
        
        eventType = np.fromfile(f, np.dtype('<u1'), 1) #always equal to 4, discard
        timestamps[currentSpike] = np.fromfile(f, np.dtype('<i8'), 1)
        software_timestamp = np.fromfile(f, np.dtype('<i8'), 1)
        source[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        numChannels = np.fromfile(f, np.dtype('<u2'), 1)
        numSamples = np.fromfile(f, np.dtype('<u2'), 1)
        sortedId[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        electrodeId = np.fromfile(f, np.dtype('<u2'),1)
        channel = np.fromfile(f, np.dtype('<u2'),1)
        color = np.fromfile(f, np.dtype('<u1'), 3)
        pcProj = np.fromfile(f, np.float32, 2)
        sampleFreq = np.fromfile(f, np.dtype('<u2'),1)
        
        waveforms = np.fromfile(f, np.dtype('<u2'), numChannels*numSamples)
        wv = np.reshape(waveforms, (numSamples, numChannels))
        
        gain[currentSpike,:] = np.fromfile(f, np.float32, numChannels)
        thresh[currentSpike,:] = np.fromfile(f, np.dtype('<u2'), numChannels)
        
        recNum[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)

        #print wv.shape        
        
        for ch in range(numChannels):
            spikes[currentSpike,:,ch] = (np.float64(wv[:,ch])-32768)/(gain[currentSpike,ch]/1000)
        
        currentSpike += 1
        
    data['spikes'] = spikes[:currentSpike,:,:]
    data['timestamps'] = timestamps[:currentSpike]
    data['source'] = source[:currentSpike]
    data['gain'] = gain[:currentSpike,:]
    data['thresh'] = thresh[:currentSpike,:]
    data['recordingNumber'] = recNum[:currentSpike]
    data['sortedId'] = sortedId[:currentSpike]

    return data
