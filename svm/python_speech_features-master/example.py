#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
print rate,sig
mfcc_feat = mfcc(sig,rate)
print len(mfcc_feat)
d_mfcc_feat = delta(mfcc_feat, 2)
print d_mfcc_feat
fbank_feat = logfbank(sig,rate)
print (fbank_feat)
print(fbank_feat[1:3,:])
