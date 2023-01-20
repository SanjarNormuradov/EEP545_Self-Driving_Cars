#!/usr/bin/env python

import rospy
import numpy as np
from threading import Lock
import matplotlib.pyplot as plt


'''
  Provides methods for re-sampling from a distribution represented by weighted samples
'''
class ReSampler:
  '''
    Initializes the resampler
    particles: The particles to sample from
    weights: The weights of each particle
    state_lock: Controls access to particles and weights
  '''
  def __init__(self, particles, weights, state_lock=None):
    self.particles = particles 
    self.weights = weights
    
    # For speed purposes, you may wish to add additional member variable(s) that 
    # cache computations that will be reused in the re-sampling functions
    self.num_particles = self.particles.shape[0]
    
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
  

  '''
    Performs independently, identically distributed in-place sampling of particles
  '''
  def resample_naiive(self):
    self.state_lock.acquire()

    self.particles[:, 0] = np.random.choice(self.particles[:, 0], size=self.num_particles, p=self.weights)

    self.state_lock.release()
  

  '''
    Performs in-place, lower variance sampling of particles
    (As discussed on pg 110 of Probabilistic Robotics)
  '''
  def resample_low_variance(self):
    self.state_lock.acquire()
    
    M = self.num_particles
    new_particles = np.zeros(self.particles.shape)
    r = np.random.uniform(0.0, 1.0 / M)
    c = self.weights[0]
    i, j = 0, 0
    for m in range(M):
      U = r + m * 1.0 / M
      while U > c:
        i += 1
        c += self.weights[i]
      new_particles[j] = self.particles[i]
      j += 1
    
    self.particles[:] = new_particles[:]
    self.state_lock.release()
     