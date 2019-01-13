# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging

from flask import Flask, request
import pickle
app = Flask(__name__)


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


@app.route('/abc', methods = ['POST'])
def hello():

	input = request.files["aud"]
	print(input)

	############ kid or adult ###################
	model = pickle.load(open("/home/reshubisht/finalized_model.sav", 'rb'))
	test_data = data_trans(input)
	pred= model.predict(test_data)
	print(pred, "prediction")



##################### appropriate??#############################

	if pred=="child":
		from sklearn.feature_extraction.text import TfidfVectorizer
		tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))#, stop_words='english')
		xt = tfidf.transform([input]).toarray()
		model2 =  pickle.load(open("/home/reshubisht/apt.sav", 'rb'))
		content=model2.predict(xt)
		if content== 1 or content==0 :  #offensive
			print("yes")
		
	   ## adult and non offensive

	else:
		print("no")

		return 'Hello World!'

	@app.errorhandler(500)
	def server_error(e):
		logging.exception('An error occurred during a request.')
		return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

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


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
