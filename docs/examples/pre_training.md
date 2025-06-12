# GridFM Pre-Training Workflow

Here we exploit the previously generated synthetic data and augment AC power flow data of (a) publicly available reference topology with a single load-profile. Thus, the overall workflow consists of:

- The first steps are data-related: We need to normalize simulation data and convert the network and power flow solution into a pytorch geometric graph representation for further processing
- Data Loader then loads the data for training
- Then some of the data-features is masked to challenge the model to reconstruct them.
- Then an autoencoder based model is trained by reconstructing the masked features. As a loss the standard "means square/absolute" error is used together with a physics informed loss, based on node-wise power balance equations (what comes in needs to get out...or be absorbed).
