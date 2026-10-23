using EarthSciIntegrator
using JSON
using Test

const FIXTURES = joinpath(@__DIR__, "..", "..", "..", "tests", "conformance", "run-configs", "valid")

@testset "EarthSciIntegrator" begin
    @testset "run-config fixtures target this spec version" begin
        files = filter(endswith(".json"), readdir(FIXTURES; join = true))
        @test !isempty(files)
        for f in files
            cfg = JSON.parsefile(f)
            @test VersionNumber(cfg["esm_integrator"]) == EarthSciIntegrator.SPEC_VERSION
        end
    end
end
