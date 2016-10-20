'''
Submit jobs to grid with random search over hyperparameters.

Adrian Benton
10/9/2016
'''

import os, sys, time

import numpy as np

if len(sys.argv) > 1:
  # initialize
  seed = int(sys.argv[1])
  np.random.seed(seed)
else:
  np.random.seed(int(time.time()))

qsubFmt = 'qsub -q all.q -cwd trainEmbeddingsJob.sh "%s" %d %f %e %e "%s" %d'

inDim    = 1000
rcov     = 1.e-6
vnameStr = ' '.join(['ego_text', 'mention_text', 'friend_text',
                     'follower_text', 'friend_net', 'follower_net'])
numEpochs  = 200
numSubJobs = 20

for j in range(numSubJobs):
  # Network dimensions on log-scale -- from 10 to 1000
  #hiddenDim = int(2**(1. + np.random.random()*9.))
  #outDim    = int(2**(1. + np.random.random()*10.))
  #k = int(2**(1. + np.random.random()*(np.log2(float(outDim)) - 1) ))
  
  hiddenDim = np.random.randint(10, 1001)
  outDim    = np.random.randint(10, 1001)
  k         = np.random.randint(10, outDim)
  
  architecture = [[inDim, hiddenDim, outDim] for i in range(6)]
  archStr = str(architecture)
  
  #l1 = np.random.choice(np.asarray([0.0, 1.e-8, 1.e-6, 1.e-4, 1.e-2]))
  l1 = np.random.choice(np.asarray([0.]))
  #l2 = np.random.choice(np.asarray([1.e-8, 1.e-6, 1.e-4, 1.e-2]))
  l2 = np.random.choice(np.asarray([0.0005]))
  
  qsubString = qsubFmt % (archStr, k, rcov, l1, l2, vnameStr, numEpochs)
  
  print(qsubString)
  
  os.system(qsubString)
  #os.system(qsubString.replace('qsub -q all.q -cwd', 'sh '))
  
  time.sleep(1)
