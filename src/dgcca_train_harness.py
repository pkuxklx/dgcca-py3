'''
Training harness for dgcca.  Apply to learning Twitter user representations.

Adrian Benton
10/7/2016
'''

import argparse, gzip, json, os

import numpy as np
import theano.tensor as T

from wgcca import ldViews
from dgcca import DGCCAArchitecture, LearningParams, DGCCA
import opt

def learnModel(prepTrainPath, prepTunePath, outputPath, modelPath, lcurveLogPath, arch, lparams, vnames, warmstart):
  '''
  Trains model and saves embedding for lowest tuning reconstruction error model.
  '''
  
  if warmstart and os.path.exists(modelPath):
    # Load model from file and continue updating weights
    model = DGCCA.load(modelPath)
    
    print ('Loaded partially-trained model from %s' % (modelPath))
  else:
    model = DGCCA(arch, lparams, vnames)
    model.build()
    
    # Update learning parameters
    if lparams is not None:
      model.lparams = lparams
  
  print('Loading training views')
  d = np.load(prepTrainPath)
  trainViews       = [np.float32(v) for v in d['views']]
  trainMissingData = np.float32(d['K'])
  
  print('Loading tune views')
  d = np.load(prepTunePath)
  tuneViews       = [np.float32(v) for v in d['views']]
  tuneMissingData = np.float32(d['K'])
  
  model.learn(trainViews, tuneViews, trainMissingData, tuneMissingData, outputPath, modelPath, lcurveLogPath, calcGMinibatch=False)
  
  return model

def prepData(inPath, prepTrainPath, prepTunePath, viewsToKeep, maxRows=-1):
  '''
  Prepares views -- saves them in numpy compressed format.  Blots out
  view+examples with all zeroes, and splits off 10% of the data for tuning.
  
  :type inPath: string
  :param inPath: views in tab-separated format
  
  :type prepTrainPath: string
  :param prepTrainPath: where to writing train data
  
  :type prepTunePath: string
  :param prepTunePath: where to write tune data
  
  :type viewsToKeep: [ int ]
  :param viewsToKeep: indices of views to keep
  
  :type maxRows: int
  :param maxRows: maximum number of rows of your data to keep
  '''
  
  # Train on all views by default
  if viewsToKeep is None:
    f = gzip.open(inPath, 'rt') if inPath.endswith('.gz') else open(inPath, 'r')
    ln = f.readline()
    numViews = int((len(ln.split('\t')) - 1)/2)
    f.close()
    
    viewsToKeep = [v for v in range(numViews)]
  
  ids, views = ldViews(inPath, viewsToKeep, maxRows=maxRows)
  
  if maxRows > 0:
    ids = ids[:maxRows]
    views = [v[:maxRows,:] for v in views]
  
  tuneCutoff = int(0.9*len(ids))
  trainIds = ids[:tuneCutoff]
  tuneIds  = ids[tuneCutoff:]
  
  trainViews = [v[:tuneCutoff,:] for v in views]
  tuneViews  = [v[tuneCutoff:,:] for v in views]
  
  trainK = np.zeros((len(trainIds), len(trainViews)))
  tuneK  = np.zeros((len(tuneIds), len(tuneViews)))
  
  for i in range(len(trainViews)):
    trainK[:,i] = 1.*(np.abs(trainViews[i]).sum(axis=1) > 0.0)
    tuneK[:,i]  = 1.*(np.abs(tuneViews[i]).sum(axis=1) > 0.0)
  
  np.savez_compressed(prepTrainPath, views=trainViews, ids=trainIds, K=trainK)
  np.savez_compressed(prepTunePath,  views=tuneViews,  ids=tuneIds,  K=tuneK)

def embeddingsToTsv(inEmbeddingDir, outTsvDir, idPath='/export/projects/abenton/multiviewTweetRepresentations/data/dgcca_10-7-2016/user_6views.full.ids.npz'):
  '''
  Convert embeddings to TSV format (formatted for eval scripts.)
  '''
  
  ps = [p for p in os.listdir(inEmbeddingDir) if p.endswith('.npz')]
  
  d = np.load(idPath)
  ids = d['arr_0']
  
  if not os.path.exists(outTsvDir):
    os.mkdir(outTsvDir)
  
  for pIdx,p in enumerate(ps):
    inPath  = os.path.join(inEmbeddingDir, p)
    outPath = os.path.join(outTsvDir, p.replace('.npz', '.tsv.gz'))
    
    embeddings = np.load(inPath)['G']
    
    outFile = gzip.open(outPath, 'wt')
    for i,row in zip(ids, embeddings):
      outFile.write('%d\t%s\n' % (i, ' '.join([str(r) for r in row])))
    
    outFile.close()
    
    print ('Converted %d/%d: %s' % (pIdx, len(ps), p))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, required=True, help='tab-separated view data')
  parser.add_argument('--preptrain', default=None, required=True,
                      help='compressed numpy file with training view data')
  parser.add_argument('--preptune', default=None, required=True,
                      help='small number of examples as tuning data')
  parser.add_argument('--maxexamples', default=-1, type=int,
                      help='number of rows to keep from training set')
  parser.add_argument('--output', default=None,
                      help='path where to save embeddings for each example')
  parser.add_argument('--model', default=None,
                      help='where to save model with lowest tune reconstruction error')
  parser.add_argument('--lcurvelog', default=None,
                      help='where to save performance per epoch')
  parser.add_argument('--arch', default=None, help='architecture for each view network, given as list of lists of layer widths')
  parser.add_argument('--k', default=None, type=int, required=True,
                      help='dimensionality of embeddings')
  parser.add_argument('--truncparam', default=1000, type=int, required=False,
                      help='rank of low-rank approximation to view matrices (for GCCA) -- default: 1000')
  #parser.add_argument('--keptviews', default=None,
  #                    type=int, nargs='+',
  #                    help='indices of views to learn model over -- defaults to using all views')
  parser.add_argument('--weights', default=None,
                      type=float, nargs='+',
                      help='how much to weight each view in the WGCCA objective -- defaults to equal weighting')
  parser.add_argument('--rcov', default=1.e-8,
                      type=float, nargs='+',
                      help='how much regularization to add to each view\'s covariance matrix.')
  #parser.add_argument('--learningRate', default=0.01,
  #                    type=float, help='initial learning rate')
  parser.add_argument('--batchSize', default=1024,
                      type=int, help='minibatch size')
  parser.add_argument('--epochs', default=200,
                      type=int, help='number of epochs to run for')
  parser.add_argument('--valfreq', default=1,
                      type=int, help='number of epochs to evaluate reconstruction error')
  parser.add_argument('--l1', default=0.0,
                      type=float, help='l1 regularization on all weights')
  parser.add_argument('--l2', default=5.e-4,
                      type=float, help='l2 regularization on all weights')
  parser.add_argument('--vnames', default=None, nargs='+',
                      help='what to name each view')
  parser.add_argument('--opt', default='{"type":"adam","params":{}}',
                      help='JSON representation of optimizer.  Look to jsonToOpt in opt.py')
  parser.add_argument('--activation', default='relu',
                      choices=['linear', 'relu', 'sigmoid', 'tanh'],
                      help='activation function to use in hidden layers')
  parser.add_argument('--warmstart', default=False, action='store_true',
                      help='if model file already exists, loads it from file then' +
                           'continues training where it left off')
  args = parser.parse_args()
  
  netArch = json.loads(args.arch)
  rcov    = args.rcov
  gccaDim = args.k
  
  if (args.output is None) and (args.model is None):
    print ('Either --model or --output must be set...')
  else:
    if not (os.path.exists(args.preptrain) and os.path.exists(args.preptune)):
      prepData(args.input, args.preptrain, args.preptune, args.keptviews, args.maxexamples)
    else:
      print('Data already prepared, skipping over preprocessing.')
    
    if args.activation == 'linear':
      activationFn = lambda x: x # linear
    elif args.activation == 'relu':
      activationFn = T.nnet.relu
    elif args.activation == 'sigmoid':
      activationFn = T.nnet.sigmoid
    elif args.activation == 'tanh':
      activationFn = T.tanh
    else:
      print('Unknown activation function %s, defaulting to ReLU' % (args.activation))
      activationFn = T.nnet.relu
    
    # Make sure we can build the optimizer
    optimizer = opt.jsonToOpt(args.opt)
    
    weights = [np.float32(w) for w in args.weights] if (args.weights is not None) else [np.float32(1.0) for r in rcov]
    
    arch    = DGCCAArchitecture(netArch, args.k, args.truncparam, activationFn)
    lparams = LearningParams(rcov=rcov, viewWts=args.weights, l1=args.l1, l2=args.l2,
                             batchSize=args.batchSize, epochs=args.epochs,
                             optStr=args.opt, valFreq=args.valfreq)
    
    learnModel(args.preptrain, args.preptune, args.output,
               args.model, args.lcurvelog, arch, lparams,
               args.vnames, args.warmstart)
  
