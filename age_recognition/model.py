import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

import aubio
import numpy as np
import wave
from aubio import source, pitch
import glob
import sys
from aubio import source, pvoc, mfcc
from numpy import vstack, zeros, diff
from python_speech_features import logfbank
from python_speech_features import fbank
import scipy.io.wavfile as wav


def model_prep():
    df=pd.read_csv('/home/reshubisht/original_data.csv')
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    filename = '/home/reshubisht/finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def data_trans(voice):

    downsample = 1
    samplerate = 44100 // downsample
    win_s = 4096 // downsample  # fft size
    hop_s = 512 // downsample  # hop size
    s = aubio.source(voice)
    samplerate = s.samplerate
    tolerance = 0.8
    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)
    total_frames = 0
    pitches = []
    confidences = []
    p = 0
    while True:
        samples, read = s()
        pitch1 = pitch_o(samples)[0]
        # pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        # if confidence < 0.8: pitch = 0.
        # print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
        pitches += [pitch1]
        confidences += [confidence]
        total_frames += read
        p = p + pitch1
        if read < hop_s: break
    mean_pitch = [p / len(pitches)]

    ######################################################################################
    s = aubio.source(voice)
    n_filters = 40  # must be 40 for mfcc
    n_coeffs = 13
    samplerate = 0
    win_s = 512
    hop_s = win_s
    mode = "default"
    samplerate = 0
    samplerate = s.samplerate
    p = pvoc(win_s, hop_s)
    m = mfcc(win_s, n_filters, n_coeffs, samplerate)
    mfccs = zeros([n_coeffs, ])
    frames_read = 0
    while True:
        samples, read = s()
        spec = p(samples)
        mfcc_out = m(spec)
        mfccs = vstack((mfccs, mfcc_out))
        frames_read += read
        if read < hop_s: break

    mfcc_result = np.mean(mfccs, axis=0)
    print(mfcc_result)

    #####################################################################
    (rate, sig) = wav.read(voice)
    logfbank_feat = logfbank(sig, rate, nfilt=10)
    fbank_result = np.mean(logfbank_feat, axis=0)

    final = mean_pitch + list(mfcc_result) + list(fbank_result)
    loaded_model = pickle.load(open(filename_as_in_google_cloud, 'rb'))
    result = loaded_model.predict(data)

    return result
