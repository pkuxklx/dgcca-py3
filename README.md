# dgcca
Python Implementation of Deep Generalized Canonical Correlation Analysis as described in 
*Adrian Benton, Huda Khayrallah, Biman Gujral, Drew Reisinger, Sheng Zhang, and Raman Arora. Deep Generalized Canonical Correlation Analysis.*  https://arxiv.org/abs/1702.02519

Tested with:

+ Python 3.5.2
+ theano 0.8.2
+ scipy 0.17.0, numpy 1.10.4
+ matplotlib 1.5.1 (for test suite)

To run the test suite:

    cd src
    python dgcca_test.py

This should take a few minutes to run on a personal laptop, time mostly spent compiling theano computation graphs.  Pdfs of synthetic data described in *Deep Generalized Canonical Correlation Analysis* are written under "test/" (original views, linear GCCA, and deep GCCA embeddings)

An example script for training a DGCCA model (using 3 different optimizers in series):

    cd src
    sh trainEmbeddingsJob_sample_grid.sh

* Input format can be grokked from: `resources/sample_wgcca_input.tsv.gz`
* DGCCA model saved to: `test/embedding_avg_dgcca_act=relu_k=100_rcov=0.000001_arch=_l1=0.0001_l2=0.01.model.npz`
* DGCCA embeddings saved to: `test/embedding_avg_dgcca_act=relu_k=100_rcov=0.000001_arch=_l1=0.0001_l2=0.01.embedding.npz`

Key
----
* `opt.py`   -- Optimizers, update network weights given current parameter values and gradient
* `mlp.py`   -- Definition of view-specific feedforward networks
* `wgcca.py` -- Weighted linear GCCA implementation.  Used as subroutine to compute gradient.
* `dgcca.py` -- Definition of full DGCCA model
* `dgcca_train_harness.py` -- Command-line tool to build and train DGCCA model 
* `dgcca_test.py` -- Test suite
* `trainEmbeddingsJob.sh` -- A more flexible version of `trainEmbeddingsJob_sample_grid.sh`

Adrian Benton
Please contact *firstname dot lastname at gmail dot com* if you have any questions, suggestions, concerns, or comments.
