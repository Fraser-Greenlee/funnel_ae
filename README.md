# Funnel AE

The Funnel Transformer converted into an Autoencoder.

## ToDo
- [x] basic setup
- [x] Do a few training runs
- [ ] Add an eval step to test during training
- [ ] Try a mini model setup - 8 tokens long input, 3 blocks
- [ ] **Beat random encoding model while having 0 skip weights.**
- [ ] Run a version which just gets a randn encoding (with resonable mean + std-dev).
- - [ ] What is the mean + std-dev of this models hidden states during training?

- [ ] Use FFN to find encoding "seperators"
- - Have a FFN take a token encoding & output a new tkn enc to +/- the original encoding for the new upsampled encodings.
- - Then don't use pool_q_only in the decoder, that will prevent the new +/- encs getting through.
- - Ideally the FFN would approximate the avg delta between the original pooled encodings.
- - Then attention should recover the actual (asymetric) delta.
- - [ ] Test this makes sense
- - - [ ] Do upsampled encodings vary less than original encodings?
- - - [ ] Is attention mostly merging adjacent tokens?

- [ ] Try new skip con w schedule
- - `block | finish by, 0 | 100, 1 | 200, 2 | 300, 3 | 400, 4 | 500, 5 | 600`

- [ ] Checkout papers that worked on similar.
- - [ ] NVAE (I'm doing similar but for text)
- - [ ] Anything doing upsampling & refining encodings.

- [ ] Run basic check runs.
- - [ ] Try with fewer blocks (so less compressed).
- - [ ] run without pool_q_only

- Old ToDos
- [ ] Add training metrics
- - [ ] add eval metrics to ensure sensible performance
- - - [ ] Plot some of the metrics found in https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard#scalars&tagFilter=lm%20loss
- [ ] Inspect model checkpoints
- - - [ ] PCA number of components per hidden state
- - - [ ] Visualise latent space of each token embedding for a given inference (see that paper which did latent space out of distribution sampling)
- - - [ ] check for plausible latent math
- [ ] Improve performance
- - [ ] Experiment with a range of skip connection weights
- - [x] Allow Encoder & Decoder to share weights.
- - [ ] Experiment with different model sizes.
- [ ] share the model
