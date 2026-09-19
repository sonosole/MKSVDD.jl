module MKSVDD

using Distances
using JuMP
using COSMO

import LinearAlgebra: diag, lmul!

include("rules.jl")
export Mercer
export CondNegDefDist
export ispsd

include("SVDD.jl")
export SVDD
export radius, radius², nsvs, alphas
export absratio, abs2ratio, sqdiff, svddprob

include("SVDD-SMO.jl")
export smosvdd

include("SVDD-Optimizer.jl")
export svdd, svddlabel


end # module MKSVDD
