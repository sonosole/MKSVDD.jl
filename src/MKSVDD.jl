module MKSVDD


using Distances
using JuMP
using COSMO

import LinearAlgebra: diag

include("kernel/inc.jl")
include("SVDD.jl")


end # module MKSVDD
