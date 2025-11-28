# -*- coding: utf-8 -*-
import os, struct 
import numpy as np
from scipy import signal as sg

from enum import Enum

# --------------------------------------------------------------------------

class Shape(Enum):
    SIN, SQU, SAW = 1,2,3

# --------------------------------------------------------------------------    
    
class Oscillator(object):

    # ----------------------------------------------------------------------
    
    def __init__(self, shape, fs=100, f=5):
        """
        shape -- the waveshape
        fs -- sample rate
        f -- frequency
        """
        self.shape = shape
        self.fs = fs
        self.f = f

    # ----------------------------------------------------------------------

    def get_samples(self, offs=0, ampl=1):
        """
        Get the sequence of samples of given waveform
        https://github.com/makermovement/3.5-Sensor2Phone
        """
        y = 0
        x = np.arange(self.fs*5000)
                
        if ( self.shape == Shape.SIN ):
            #y = [ offs+ampl*np.sin(2*np.pi*self.f * (i/self.fs)) for i in x]
            y = offs + ampl * np.sin(2 * np.pi * self.f * x / self.fs )
        if ( self.shape == Shape.SQU ):
            y = offs + ampl * sg.square(2 * np.pi * self.f * x / self.fs )
        if ( self.shape == Shape.SAW ):
            y = offs + ampl * sg.sawtooth(2 * np.pi * self.f * x / self.fs )
                  
        return y

def save_wav(file_name, samples, sample_rate=44100.0):
    import scipy.io.wavfile
    scipy.io.wavfile.write(file_name, int(sample_rate), np.array(samples).astype(np.int16))

def play(file_name):
    import subprocess
    # C:\Windows\System32\rundll32.exe
    subprocess.call(['rundll32', 'C:\Program Files\Windows Photo Viewer\PhotoViewer.dll', 'ImageView_Fullscreen', file_name])

if __name__ == '__main__':
    o = Oscillator(Shape.SIN) #,fs=44100,f=1000
    samples = o.get_samples(offs=0, ampl=1)

    file_name = os.getcwd() + '/test.wav'
    save_wav(file_name, samples*1000)
    play(file_name)
    print(samples)
    
'''
pip install scipy
'''