'''
Test suite for deep GCCA.  Plots projected views for deep GCCA and vanilla GCCA.

Adrian Benton
9/14/2016
'''

import os
import unittest
from functools import reduce

from mlp   import MLPWithLinearOutput
from dgcca import DGCCAArchitecture, LearningParams, DGCCA
from wgcca import WeightedGCCA

import theano

import numpy as np
import scipy
import scipy.io
import scipy.linalg

import seaborn as sns

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

import theano.tensor as T

class TestWeightedGCCA(unittest.TestCase):
  def setUp(self):
    self.outDir = '../test'
    if not os.path.exists(self.outDir):
      os.mkdir(self.outDir)
    
    ### Generate sample data with 3 views and two classes:
    ### 1) Two concentric circles
    ### 2) Two parabolas with different intercept and quadratic coefficient
    ### 3) Two concentric circles with classes reversed and greater variance
    self.N  = 400 # Number of examples
    self.F1 = 2   # Number of features in view 1
    self.F2 = 2   
    self.F3 = 2   
    self.k  = 2   # Number of latent features
    
    # First half of points belong to class 1, second to class 2
    G = np.zeros( ( self.N, self.k ) )
    
    G[:int(self.N/2),0] = 1.0
    G[int(self.N/2):,1] = 1.0
    self.classes = ['Class1' for i in range(int(self.N/2))] + ['Class2' for i in range(int(self.N/2))]
    
    # Each class lies on a different concentric circle
    rand_angle = np.random.uniform(0.0, 2.0 * math.pi, (self.N, 1) )
    rand_noise = 0.1 * np.random.randn(self.N, self.k)
    circle_pos = np.hstack( [np.cos(rand_angle), np.sin(rand_angle)])
    radius     = G.dot(np.asarray( [[1.0], [2.0]] )).reshape( (self.N, 1) )
    self.V1    = np.hstack([radius, radius]) * circle_pos + rand_noise
    
    # Each class lies on a different parabola
    rand_x     = np.random.uniform(-3.0, 3.0, (self.N, 1) )
    rand_noise = 0.1 * np.random.randn(self.N, self.k)
    intercepts = G.dot( np.asarray([[0.0], [1.0]])).reshape( (self.N, 1) )
    quadTerms  = G.dot( np.asarray( [[2.0], [0.5]] )).reshape( (self.N, 1) )
    self.V2    = np.hstack( [rand_x, intercepts + quadTerms * (rand_x*rand_x)]) + rand_noise
    
    # Class on inside is drawn from a gaussian, class on outside is on a concentric circle
    rand_angle = np.random.uniform(0.0, 2.0 * math.pi, (self.N, 1) )
    inner_rand_noise = 1.0 * np.random.randn(int(self.N/2), self.k) # More variance on inside
    outer_rand_noise = 0.1 * np.random.randn(int(self.N/2), self.k)
    rand_noise = np.vstack( [outer_rand_noise, inner_rand_noise] )
    circle_pos = np.hstack( [np.cos(rand_angle), np.sin(rand_angle)])
    radius     = G.dot(np.asarray( [[2.0], [0.0]] )).reshape( (self.N, 1) )
    self.V3    = np.hstack([radius, radius]) * circle_pos + rand_noise
    
    # We have no missing data
    self.K = np.ones( (self.N, 3) )
    
    # Gather into dataframes to plot
    self.views = [self.V1, self.V2, self.V3]
    dfs = []
    for v in self.views:
      df = pd.DataFrame(v, columns=['x', 'y'])
      df['Classes'] = self.classes
      dfs.append(df)
    
    # Plot to PDF
    with PdfPages(os.path.join(self.outDir, 'originalData.pdf')) as pdf:
      for viewIdx, df in enumerate(dfs):
        fig = sns.lmplot(x="x", y="y", fit_reg=False, markers=['+', 'o'], legend=False, hue="Classes", data=df).fig
        plt.legend(loc='best')
        plt.title('View %d' % (viewIdx))
        pdf.savefig()
    
  def tearDown(self):
    ''' Remove generated files '''
    pass
  
  def testSerializeModel(self):
    '''
    Can we save & load the same model?
    '''
    
    # Simple, simple model, 3 views, one nonlinear hidden layer
    viewMlpStruct = [ [2, 5, 2], [2, 5, 2], [2, 5, 2] ] 
    
    # Actual data used in ICML draft...
    d = scipy.io.loadmat('../resources/synthdata.mat')
    self.views = [d['view1'], d['view2'], d['view3']]
    self.K = np.ones((400,3))
    
    arch = DGCCAArchitecture(viewMlpStruct, 2, activation=T.nnet.relu)
    
    # Little bit of L2 regularization -- learning params from matlab synthetic experiments
    lparams = LearningParams( rcov=[0.01]*3, l1=[0.0]*3, l2=[5.e-4]*3,
                              optStr='{"type":"sgd","learningRate":0.01,"decay":1.0}',
                              batchSize=400,
                              epochs=10)
    vnames = ['View1', 'View2', 'View3']
    
    model = DGCCA(arch, lparams, vnames)
    model.build(initWeights=None, randSeed=12345)
    
    model.save(os.path.join(self.outDir, 'test.model.npz'))
    reloadedModel = DGCCA.load(os.path.join(self.outDir, 'test.model.npz'))
    
    # reloaded model should have the same weights
    origWts = model._model.getWeights()
    for vIdx, reloadedView in enumerate(reloadedModel._model.getWeights()):
      for lIdx, reloadedWts in enumerate(reloadedView):
        self.assertTrue( np.allclose(reloadedWts, origWts[vIdx][lIdx]) )
  
  def testGCCA(self):
    '''
    Linear projection of views to latent space.  Classes should not be linearly separable
    after applying GCCA, lie directly on each other.
    '''
    
    wgcca = WeightedGCCA(3, [2, 2, 2], 2, 0.01, 2)
    wgcca = wgcca.learn(self.views, self.K)
    
    G = wgcca.G
    
    df = pd.DataFrame(G, columns=['x', 'y'])
    df['Classes'] = self.classes
    
    # Plot to PDF
    with PdfPages(os.path.join(self.outDir, 'vanillaGccaData.pdf')) as pdf:
      fig = sns.lmplot(x="x", y="y", fit_reg=False, markers=['+', 'o'], legend=False, hue="Classes", data=df).fig
      plt.legend(loc='best')
      plt.title('Vanilla GCCA mapping')
      pdf.savefig()
  
  def testDGCCA(self):
    '''
    Train a simple network on these data, plot the projected views at each
    iteration.  Points should be more-or-less linearly separable after this transformation.
    '''
    
    viewMlpStruct = [ [2, 10, 10, 10, 2], [2, 10, 10, 10, 2], [2, 10, 10, 10, 2] ] # Each view has single-hidden-layer MLP with slightly wider hidden layers
    
    # Actual data used in paper plot...
    d = scipy.io.loadmat('../resources/synthdata.mat')
    self.views = [np.float32(d['view1']), np.float32(d['view2']), np.float32(d['view3'])]
    self.K = np.ones((400,3), dtype=np.float32)
    
    arch = DGCCAArchitecture(viewMlpStruct, 2, activation=T.nnet.relu)
    
    # Little bit of L2 regularization -- learning params from matlab synthetic experiments
    lparams = LearningParams( rcov=[0.01]*3, l1=[0.0]*3, l2=[5.e-4]*3,
          optStr='{"type":"adam","params":{"adam_b1":0.1,"adam_b2":0.001}}',
                              batchSize=400,
                              epochs=200)
    vnames = ['View1', 'View2', 'View3']
    
    model = DGCCA(arch, lparams, vnames)
    model.build()
    
    history = []
    
    # Plot to PDF
    with PdfPages(os.path.join(self.outDir, 'dgccaData.pdf')) as pdf:
      for epoch in range(2):
        if epoch > 0: # Want to plot random initialization first, so skip first iteration
          history.extend(model.learn(self.views, tuneViews=None, trainMissingData=self.K,
                                     tuneMissingData=None, embeddingPath=None,
                                     modelPath=None, logPath=None, calcGMinibatch=False))
        
        # Plot output layers
        outputViews = model._model.get_outputs_centered(*self.views)
        for VIdx, output in enumerate(outputViews):
          df = pd.DataFrame(output, columns=['x', 'y'])
          df['Classes'] = self.classes
          
          fig = sns.lmplot(x="x", y="y", fit_reg=False, markers=['+', 'o'],
                           legend=False, hue="Classes", data=df).fig
          plt.legend(loc='best')
          plt.title('View %d Output layer -- Epoch %d' % (VIdx, epoch*200))
          pdf.savefig()
        
        # Plot multiview embeddings
        G = model.apply(self.views, self.K)
        df = pd.DataFrame(G, columns=['x', 'y'])
        df['Classes'] = self.classes
        
        fig = sns.lmplot(x="x", y="y", fit_reg=False, markers=['+', 'o'],
                         legend=False, hue="Classes", data=df).fig
        plt.legend(loc='best')
        plt.title('Deep G -- Epoch %d' % (epoch*200))
        pdf.savefig()
  
  def testExternalGrad(self):
    '''
    GCCA gradient is backpropagated by computing a hacky pseudocost that Theano auto-diffs.
    This is a test of the hack.  Builds a simple MLP, and known cost function (2-norm of
    linear map of output layer.)
    '''
    
    F1 = 100
    F2 = 50
    F3 = 20
    
    X = T.matrix('X') # Input
    
    # Parameters
    W = theano.shared(np.random.randn(F2, F1).astype(theano.config.floatX), name='W')
    W2 = theano.shared(np.random.randn(F3, F2).astype(theano.config.floatX), name='W2')
    V = theano.shared(np.random.randn(F3, 1).astype(theano.config.floatX), name='V')
    
    Y = T.nnet.relu( T.nnet.relu( X.dot(W.T) ).dot(W2.T) )
    
    # This function will be a mystery to us...
    Zvec = Y.dot(V)
    Z = T.sum ( T.sqrt( Zvec.T.dot(Zvec) ) )
    
    trueGrad     = T.grad(Z, W)
    trueGradWrtY = T.grad(Z, Y)
    
    Gcopy = T.matrix('Gcopy')
    
    fake_cost = T.sum( Y * Gcopy )
    guessedGrad  = T.grad( fake_cost, W )
    guessedGradY = T.grad( fake_cost, Y )
    
    getY = theano.function([X], Y)
    getZ = theano.function([X], Z)
    
    getTrueGradWrtY = theano.function([X], trueGradWrtY)
    
    getGuessedGrad  = theano.function([X, Gcopy], guessedGrad)
    getGuessedGradY = theano.function([X, Gcopy], guessedGradY)
    getTrueGrad     = theano.function([X], trueGrad)
    
    for run in range(100):
      Xtest = np.random.randn(1000, F1)
      
      TG, TGwrtY = getTrueGrad(Xtest), getTrueGradWrtY(Xtest)
      Gguess, GguessY = getGuessedGrad(Xtest, TGwrtY), getGuessedGradY(Xtest, TGwrtY)
      
      assert( np.allclose(TGwrtY, GguessY, 1.e-8) )
      assert( np.allclose(TG, Gguess, 1.e-8) )

  def testSgdUpdate(self):
    '''
    Makes sure the gradient we calculate is used to take steps.
    '''
    
    np.random.seed(12345)
    sample_n_examples = 400
    sample_n_hidden = [ 50, 10, 5 ]
    sample_input = np.random.randn(sample_n_examples, sample_n_hidden[0])
    sample_externalGrad  = np.random.randn(sample_n_examples, sample_n_hidden[-1])
    
    arch = sample_n_hidden
    from opt import SGDOptimizer
    
    def getNextWts(currWts, currGrad):
      nextWts = []
      for v, (Ws, Gs) in enumerate(zip(currWts, currGrad)):
        nextWs = []
        for W, G in zip(Ws, Gs):
          nextWs.append(W + G)
        nextWts.append(nextWs)
      
      return nextWts
    
    def diffWts(wts1, wts2):
      diff = 0.0
      for v, (Ws1, Ws2) in enumerate(zip(wts1, wts2)):
        for W1,W2 in zip(Ws1, Ws2):
          diff += np.sum((W1-W2)**2.)
      
      return np.sqrt(diff)
    
    lr = 0.01
    optimizer = SGDOptimizer(learningRate=0.01)
    
    print('... building the model')
    mlp = MLPWithLinearOutput(12345, arch, T.nnet.relu, 5, optimizer,
                              L1_reg=0.0, L2_reg=0.01, vname='MLPtest')
    mlp.buildFns()
    
    sampleGrad   = mlp.calc_reg_gradient(sample_input, sample_externalGrad)
    
    # Make sure weights are being updated as we expect
    allParams = []
    allGrads  = []
    for i in range(20):
      cost = mlp.calc_total_cost(sample_input, sample_externalGrad)
      grad = mlp.calc_reg_gradient(sample_input, sample_externalGrad)
      
      #print (i, cost)
      allParams.append([p for p in mlp.getWeights()])
      
      mlp.take_step(sample_input, sample_externalGrad)
      
      allGrads.append(grad)
    
    predParams = [getNextWts(p, [lr*g for g in gs]) for p, gs in zip(allParams, allGrads)]
    diffs = [diffWts(p1, p2) for p1, p2 in zip(allParams[1:], predParams)]
    
    for diff in diffs:
      print(str(diff) + '...', end=' ')
      self.assertTrue( diff < 1.e-8 )
    print ('\n')

def main():
  unittest.main()

if __name__ == '__main__':
  main()

