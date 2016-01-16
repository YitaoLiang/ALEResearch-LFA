# ALEResearch-LFA
Research on ALE --- mainly about bpro features and its extensions.

We choose to focus on using the linear function approximation, which is a simpler yet fixed representation approach.

We aim to achieve performance comparable to DQN's (Deep Q-networks)

Bpro and its extensions exploit spatial invariance, temporal offsets and a simple way to detect sprites.

The codes in the master branch are not used to generate any research results. Instead it lays the premiliminary foundation for real research.
BproVector has the fastest codes to run BPRO feature set.
bproTime has the codes to run BPROST feature set.
Blob (branch) has the codes to run Blob-BPROST features set (blob is the name of our simple methods to detect sprites on screens).
Other braches have some other experiments (e.g we tried to implement tug-of-wash hashing)
Another intersting brach is blobThreeVersion1(and blobThreeVersion2). Instead of doing pairwise offsets, it does offsets between three pixels. Still it has spatial invariance built in.

In blob and bproTime, it has seperate functions to generate temporal offsets (one pixel comes from current screen and the other comes from the screen five frames ago  and spatial offsets (both pixels come from current screens). The fucntions are called addTimeDimensionalOffsets and addRelativeFeaturesIndices respectively. In case you only want to run Blob-PROS or Blob-PROT (or B-PROS/ B-PROT), you can commet out the corresponding functions. Both have a redundant function called addThreePointOffsetsIndices, it never gets called. 

The latest research results are available on http://arxiv.org/abs/1512.01563

Braches start with Adaptive include our codes to attempt an adaptive feature representation in ALE. The work is still in progress.


