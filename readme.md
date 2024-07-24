## Multi-Stage Face-Voice Association Learning with Keynote Speaker Diarization (ACM MM 2024)

### Introduction

This code is for HLT system for the FAME challenge in ACM MM 2024.

'pretrain/seen.pt' is our model under the seen condition.

We did not put the validation list/code instead. Now getting the test results each epoch. It can be easily added.

### Data and running

The voice data after the keynote speaker diarization can be found in this link:

Our system contains multiple training stages, this code only contains the final FAME adaption process. 

`bash run_train.sh'