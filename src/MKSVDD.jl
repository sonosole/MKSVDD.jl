module MKSVDD


using Distances
using JuMP
using COSMO

import LinearAlgebra: diag

include("kernel/inc.jl")
export ChebyK
export CityK
export CosK
export EucK
export HammingK
export JaccardK
export KLDivK
export MAEK
export MSEK
export MinkowskiK
export NRMSEK
export RBFK
export RMSEK
export kmat

include("SVDD.jl")
export SVDD
export svddlabel


end # module MKSVDD
