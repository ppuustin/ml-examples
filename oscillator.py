# -*- coding: utf-8 -*-
import struct 
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
        x = np.arange(self.fs)
                
        if ( self.shape == Shape.SIN ):
            #y = [ offs+ampl*np.sin(2*np.pi*self.f * (i/self.fs)) for i in x]
            y = offs + ampl * np.sin(2 * np.pi * self.f * x / self.fs )
        if ( self.shape == Shape.SQU ):
            y = offs + ampl * sg.square(2 * np.pi * self.f * x / self.fs )
        if ( self.shape == Shape.SAW ):
            y = offs + ampl * sg.sawtooth(2 * np.pi * self.f * x / self.fs )
        
        #with open('test.wav','wb') as f:
        #    for i in y: f.write(struct.pack('b', i))          
          
        return y
        
if __name__ == '__main__':
    o = Oscillator(Shape.SIN)
    samples = o.get_samples(offs=0, ampl=1)
    print(samples)