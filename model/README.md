# Model 
- Most of the runs in the paper are done using the default model `vnet`. They are located here: https://github.com/timothychan29475600/map2map/tree/master/map2map/models
- The model `dustnet` is used in mc-dropout. The code is in this directory (and also in the link above)
- `vnet` is also used in SWAG runs, but with a modified version of `map2map` is used. This can be found here: https://github.com/timothychan29475600/map2map/tree/master/map2map under `test_swag` and `train_swag`. The actual SWAG model is ported from https://github.com/wjmaddox/swa_gaussian.
