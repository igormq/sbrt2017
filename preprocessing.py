# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import math
import decimal
import string
import numpy as np
from unidecode import unidecode
import logging

from scipy import signal
from scipy.fftpack import dct
import librosa


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'),
                                                rounding=decimal.ROUND_HALF_UP
                                                ))


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame
    that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no
    window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))

    indices = np.tile(
        np.arange(
            0, frame_len),
        (numframes, 1)) + np.tile(
            np.arange(
                0, numframes * frame_step, frame_step), (frame_len, 1)).T

    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))
    return frames * win


def deframesig(frames, siglen, frame_len, frame_step,
               winfunc=lambda x: np.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.
    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output
    will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame
    that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no
    window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong\
    size, 2nd dim is not equal to frame_len'

    indices = np.tile(
        np.arange(
            0, frame_len), (numframes, 1)) + np.tile(
                np.arange(
                    0, numframes * frame_step, frame_step), (frame_len, 1)).T

    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0:
        siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        # add a little bit so it is never zero
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + \
                                           win + 1e-15
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an
    NxD matrix, output will be NxNFFT.
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD
    matrix, output will be NxNFFT.
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an
    NxD matrix, output will be NxNFFT.
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the
    max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features)
    containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and
    following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features)
    containing delta features. Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = np.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for
                                                                  i in
                                                                  range(N)]))
    denom = sum([2 * i * i for i in range(1, N + 1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(np.sum([n * feat[N + j + n]
                                for n in range(-1 * N, N + 1)], axis=0) /
                     denom)
    return dfeat

class Feature(object):
    """ Base class for features calculation
    All children class must implement __str__ and _call function.

    # Arguments
        fs: sampling frequency of audio signal. If the audio has not this fs,
        it will be resampled
        eps
    """

    def __init__(self, fs=16e3, eps=1e-8,
                 mean_norm=True, var_norm=True):
        self.fs = fs
        self.eps = eps

        self.mean_norm = mean_norm
        self.var_norm = var_norm

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

    def __call__(self, audio):
        """ This method load the audio and do the transformation of signal

        # Inputs
            audio:
                if audio is a string and the file exists, the wave file will
                be loaded and resampled (if necessary) to fs
                if audio is a ndarray or list and is not empty, it will make
                the transformation without any resampling

        # Exception
            TypeError if audio were not recognized

        """
        if ((isinstance(audio, str) or isinstance(audio, unicode))
            and os.path.isfile(audio)):
            audio, current_fs = librosa.audio.load(audio)
            audio = librosa.core.resample(audio, current_fs, self.fs)
            feats = self._call(audio)
        elif type(audio) in (np.ndarray, list) and len(audio) > 1:
            feats = self._call(audio)
        else:
            TypeError("audio type is not support")

        return self._standarize(feats)

    def _call(self, data):
        raise NotImplementedError("__call__ must be overrided")

    def _standarize(self, feats):
        if self.mean_norm:
            feats -= np.mean(feats, axis=0, keepdims=True)
        if self.var_norm:
            feats /= (np.std(feats, axis=0, keepdims=True) + self.eps)
        return feats

    def __str__(self):
        raise NotImplementedError("__str__ must be overrided")

    @property
    def num_feats(self):
        return self._num_feats


class FBank(Feature):
    """Compute Mel-filterbank energy features from an audio signal.

    # Arguments
        win_len: the length of the analysis window in seconds.
            Default  is 0.025s (25 milliseconds)
        win_step: the step between successive windows in seconds.
            Default is 0.01s (10 milliseconds)
        num_filt: the number of filters in the filterbank, default 40.
        nfft: the FFT size. Default is 512.
        low_freq: lowest band edge of mel filters in Hz.
            Default is 20.
        high_freq: highest band edge of mel filters in Hz.
            Default is 7800
        pre_emph: apply preemphasis filter with preemph as coefficient.
        0 is no filter. Default is 0.97.
        win_func: the analysis window to apply to each frame.
            By default hamming window is applied.
    """

    def __init__(self, win_len=0.025, win_step=0.01,
                 num_filt=40, nfft=512, low_freq=20, high_freq=7800,
                 pre_emph=0.97, win_fun=signal.hamming, **kwargs):

        super(FBank, self).__init__(**kwargs)

        if high_freq > self.fs / 2:
            raise ValueError("high_freq must be less or equal than fs/2")

        self.win_len = win_len
        self.win_step = win_step
        self.num_filt = num_filt
        self.nfft = nfft
        self.low_freq = low_freq
        self.high_freq = high_freq or self.fs / 2
        self.pre_emph = pre_emph
        self.win_fun = win_fun
        self._filterbanks = self._get_filterbanks()

        self._num_feats = self.num_filt

    @property
    def mel_points(self):
        return np.linspace(self._low_mel, self._high_mel, self.num_filt + 2)

    @property
    def low_freq(self):
        return self._low_freq

    @low_freq.setter
    def low_freq(self, value):
        self._low_mel = self._hz2mel(value)
        self._low_freq = value

    @property
    def high_freq(self):
        return self._high_freq

    @high_freq.setter
    def high_freq(self, value):
        self._high_mel = self._hz2mel(value)
        self._high_freq = value

    def _call(self, signal):
        """Compute Mel-filterbank energy features from an audio signal.
        :param signal: the audio signal from which to compute features. Should
        be an N*1 array

        Returns:
            2 values. The first is a numpy array of size (NUMFRAMES by nfilt)
            containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy,
            unwindowed)
        """

        signal = preemphasis(signal, self.pre_emph)

        frames = framesig(signal,
                          self.win_len * self.fs,
                          self.win_step * self.fs,
                          self.win_fun)

        pspec = powspec(frames, self.nfft)
        # this stores the total energy in each frame
        energy = np.sum(pspec, 1)
        # if energy is zero, we get problems with log
        energy = np.where(energy == 0, np.finfo(float).eps, energy)

        # compute the filterbank energies
        feat = np.dot(pspec, self._filterbanks.T)
        # if feat is zero, we get problems with log
        feat = np.where(feat == 0, np.finfo(float).eps, feat)

        return feat, energy

    def _get_filterbanks(self):
        """Compute a Mel-filterbank. The filters are stored in the rows, the
        columns correspond
        to fft bins. The filters are returned as an array of size nfilt *
        (nfft / 2 + 1)

        Returns:
            A numpy array of size num_filt * (nfft/2 + 1) containing
            filterbank. Each row holds 1 filter.
        """

        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((self.nfft + 1) * self._mel2hz(self.mel_points) /
                       self.fs)

        fbank = np.zeros([self.num_filt, int(self.nfft / 2 + 1)])
        for j in xrange(0, self.num_filt):
            for i in xrange(int(bin[j]), int(bin[j + 1])):
                fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in xrange(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        return fbank

    def _hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        Args:
            hz: a value in Hz. This can also be a numpy array, conversion
            proceeds element-wise.

        Returns:
            A value in Mels. If an array was passed in, an identical sized
            array is returned.
        """
        return 2595 * np.log10(1 + hz / 700.0)

    def _mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        Args:
            mel: a value in Mels. This can also be a numpy array, conversion
            proceeds element-wise.

        Returns:
            A value in Hertz. If an array was passed in, an identical sized
            array is returned.
        """
        return 700 * (10**(mel / 2595.0) - 1)

    def __str__(self):
        return "fbank"


class MFCC(FBank):
    """Compute MFCC features from an audio signal.

    # Arguments
        num_cep: the number of cepstrum to return. Default 13.
        cep_lifter: apply a lifter to final cepstral coefficients. 0 is
        no lifter. Default is 22.
        append_energy: if this is true, the zeroth cepstral coefficient
        is replaced with the log of the total frame energy.
        d: if True add deltas coeficients. Default True
        dd: if True add delta-deltas coeficients. Default True
        norm: if 'cmn' performs the cepstral mean normalization. elif 'cmvn'
        performs the cepstral mean and variance normalizastion. Default 'cmn'
    """

    def __init__(self, num_cep=13, cep_lifter=22, append_energy=True,
                 d=True, dd=True, **kwargs):

        super(MFCC, self).__init__(**kwargs)

        self.num_cep = num_cep
        self.cep_lifter = cep_lifter
        self.append_energy = append_energy
        self.d = d
        self.dd = dd
        self._num_feats = (1 + self.d + self.dd) * self.num_cep

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

    def _call(self, signal):
        """Compute MFCC features from an audio signal.

        Args:
            signal: the audio signal from which to compute features. Should be
            an N*1 array

        Returns:
            A numpy array of size (NUMFRAMES by numcep) containing features.
            Each row holds 1 feature vector.
        """
        feat, energy = super(MFCC, self)._call(signal)

        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :self.num_cep]
        feat = self._lifter(feat, self.cep_lifter)

        if self.append_energy:
            # replace first cepstral coefficient with log of frame energy
            feat[:, 0] = np.log(energy + self.eps)

        if self.d:
            d = delta(feat, 2)
            feat = np.hstack([feat, d])

            if self.dd:
                feat = np.hstack([feat, delta(d, 2)])

        return feat

    def _lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra.

        This has the effect of increasing the magnitude of the high frequency
        DCT coeffs.

        Args:
            cepstra: the matrix of mel-cepstra, will be numframes * numcep in
            size.
            L: the liftering coefficient to use. Default is 22. L <= 0 disables
            lifter.
        """
        if L > 0:
            nframes, ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L / 2) * np.sin(np.pi * n / L)
            return lift * cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

    def __str__(self):
        return "mfcc"


class SimpleCharParser(object):
    """ Class responsible to map any text in a certain character vocabulary

    # Arguments
        mode: Which type of vacabulary will be generated. Modes can be
        concatenated by using pipeline '|'
            'space' or 's': accepts space character
            'accents' or 'a': accepts pt-br accents
            'punctuation' or 'p': accepts punctuation defined in
            string.punctuation
            'digits': accepts all digits
            'sensitive' or 'S': characters will be case sensitive
            'all': shortcut that enables all modes
    """

    def __init__(self):

        self._vocab, self._inv_vocab = self._gen_vocab()

    def map(self, txt, sanitize=True):
        if sanitize:
            label = np.array([self._vocab[c] for c in self._sanitize(txt)],
                             dtype='int32')
        else:
            label = np.array([self._vocab[c] for c in txt], dtype='int32')

        return label

    def imap(self, labels):
        txt = ''.join([self._inv_vocab[l] for l in labels])

        return txt

    def _sanitize(self, text):
        # removing duplicated spaces
        text = ' '.join(text.split())

        # removing digits
        text = ''.join([c for c in text if not c.isdigit()])

        # removing accents
        text = unidecode(text)

        # removnig punctuations
        text = text.translate(
            string.maketrans("-'", '  ')).translate(None,
                                                    string.punctuation)

        # remove uppercase
        text = text.lower()

        return text

    def is_valid(self, text):
        # verify if the text is valid without sanitization
        try:
            _ = self.map(text, sanitize=False)
            return True
        except KeyError:
            return False

    def _gen_vocab(self):

        vocab = {chr(value + ord('a')): (value)
                 for value in xrange(ord('z') - ord('a') + 1)}

        vocab[' '] = len(vocab)

        inv_vocab = {v: k for (k, v) in vocab.iteritems()}

        # Add blank label
        inv_vocab[len(inv_vocab)] = '<b>'

        return vocab, inv_vocab

    def __call__(self, _input):
        return self.map(_input)
