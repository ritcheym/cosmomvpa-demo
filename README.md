# Introduction to CosmoMVPA
Download [CoSMoMVPA version 1.1.0](https://github.com/CoSMoMVPA/CoSMoMVPA/releases/tag/v.1.1.0)

## Demo dataset
The demo dataset can be downloaded [here](https://drive.google.com/open?id=0B3ALgJ4N0EPWbGl5QmNQRl9uR0E) (BC sign-in required, for now)

### Design features
- 6 runs of functional data from a single awesome subject
- Task design: Category (face, scene) x Fame (famous, non-famous) x Repetition (first, second)
  - Famous/non-famous scenes and faces randomly intermixed
  - First repetitions are in runs 1-3
  - Second repetitions are in runs 4-6
- No overt task during encoding
- Item memory and familiarity with famous faces/scenes were assessed afterward
- Fixed ISI - 8 seconds

### Processing
The functional data have already been preprocessed, including realignment and normalization to MNI space. All original and preprocessed data can be found in `data`

### Modeling
Two models have been run:

1. A standard model implementing the 2x2x2 design described above: `standard_model`
2. A single-trial model with a regressor for each individual trial onset: `singletrial_model`
   - To facilitate the demo, the single-trial betas have been copied to `singletrial_betas` & renamed according to their trial information.  

You could use either model for MVPA-- the standard model will give you more stable beta parameters, but the single-trial model will give you more samples (observations) to use. The single-trial model also allows for item-specific analyses, so I'll be using it for the demo. 

Note that because the ISI is fixed and medium-long (8s), we could also use unmodeled timepoints from 4-6s post-onset (but we won't for the demo).

## Multi-voxel pattern classification



## Representational similarity


