
Alignment methods inside biocore.py:
- align_leftmost_edge
- huang_normalization
- shift_to_CoM
Suggestion: switch to the mask computed by fingerfocus instead of leemask as Caroline and I had some problems with leemask
We added the mask as return of fingerfocus and use it further

Evaluation of alignment methods
 - python3 alignment_evaluation.py
    - returns a list of (Hamming dist(a,b), nr_of_1s(a), nr_of_1s(b))
    - prerequisite: folder path with png caputers
    - each method tested should be called, e.g., either in biocore preprocess or in this script after feature Extraction

Computation of mean and variance:
- python3 meanandvariance.py
    - returns the mean and variance for both biometric hash coming from same and different persons and entropy
    - prerequisite: folder path with biometric hashes
    - to compute the biometric hash: python3 hashing.py
          - prerequisite: folder path with png captures
          - output: given folder path, adds hash.npy for each pair of images coming from same person (attempt1 the model, attempt2 a probe)

Also, during the project research I found 2 other feature extraction methods in IDIAP's bob library which I've added in extraction.py
one can simply use these at return of extract_features (biocore.py):
-  wide_line_detector(*preprocess(image,mask))
-  repeated_line_tracking(*preprocess(image,mask))

Comp-FE-master: Fuzzy Extractor construction from https://github.com/benjaminfuller/CompFE
