function [bsdata, indices] = stationary_bootstrap(data, B, w)
% Replaces the MFE Toolbox sampler for the identical-index run: it hands back the resamples this
% library drew, so any difference left comes from the statistic and not from the draws.
global OLAB_INDICES
indices = OLAB_INDICES;
bsdata = data(indices);
end
