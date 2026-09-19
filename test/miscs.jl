@testset "Basic functions" begin
    ker(x::AbstractArray, y::AbstractArray) = abs.(x' * y)
    R² = 9.0
    wᵀKw = 3.0
    n = 2 # support vectors, 1 row matrix
    w = rand(1, n)
    fdims = 7
    svecs = rand(fdims, n)
    model = SVDD{Float64,1}(R², wᵀKw, w, svecs, ker)
    @test radius²(model) == radius(model)^2
    x = rand(fdims, 10)
    @test all(abs(model, x).^2 .≈ abs2(model, x))
    @test all(absratio(model, x).^2 .≈ abs2ratio(model, x))
    @test all(0 .≤ svddprob(model, x, type="gaussian") .≤ 1)
    @test all(0 .≤ svddprob(model, x, type="laplace") .≤ 1)
    @test all(0 .≤ svddprob(model, x, type="triangle") .≤ 1)
    @test all(0 .≤ svddprob(model, x, type="sqtriangle") .≤ 1)
    @test all(0 .≤ svddprob(model, x, type="dirac") .≤ 1)
end;
