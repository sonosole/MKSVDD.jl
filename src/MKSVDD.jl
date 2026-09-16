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
export radius, alphas, svddprob

include("SVDD-SMO.jl")
export smosvdd

include("SVDD-Optimizer.jl")
export svdd, svddlabel


end # module MKSVDD
