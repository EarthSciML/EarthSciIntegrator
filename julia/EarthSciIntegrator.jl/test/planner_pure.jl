# The planner's parts that need no EarthSciAST (M2 stage 1): canonical JSON
# (spec/plan.md §2.1), the run-config schema check (§4 step 1), and the band
# calculus of pass 2 (§5.2.3–§5.2.4). The passes over a model are tested in
# test/planner/ (the EarthSciAST extension), see runtests.jl.

using EarthSciIntegrator: BandEntry, ecma_number, jcs, schema_errors, run_config_schema, classify_bands, reduce_offset,
    raw_offsets, Axes

const REPO = joinpath(@__DIR__, "..", "..", "..")
const CONF = joinpath(REPO, "tests", "conformance")
const PFIX = joinpath(CONF, "plans", "fixtures")

@testset "planner, without EarthSciAST (M2 stage 1)" begin
    @testset "RFC 8785 numbers" begin
        # RFC 8785 Appendix B, by bit pattern.
        for (bits, s) in [
            0x0000000000000000 => "0", 0x8000000000000000 => "0", 0x0000000000000001 => "5e-324",
            0x8000000000000001 => "-5e-324", 0x7fefffffffffffff => "1.7976931348623157e+308",
            0xffefffffffffffff => "-1.7976931348623157e+308", 0x4340000000000000 => "9007199254740992",
            0xc340000000000000 => "-9007199254740992", 0x4430000000000000 => "295147905179352830000",
            0x44b52d02c7e14af5 => "9.999999999999997e+22", 0x44b52d02c7e14af6 => "1e+23",
            0x44b52d02c7e14af7 => "1.0000000000000001e+23", 0x444b1ae4d6e2ef4e => "999999999999999700000",
            0x444b1ae4d6e2ef4f => "999999999999999900000", 0x444b1ae4d6e2ef50 => "1e+21",
            0x3eb0c6f7a0b5ed8c => "9.999999999999997e-7", 0x3eb0c6f7a0b5ed8d => "0.000001",
            0x41b3de4355555553 => "333333333.3333332", 0x41b3de4355555554 => "333333333.33333325",
            0x41b3de4355555555 => "333333333.3333333", 0x41b3de4355555556 => "333333333.3333334",
            0x41b3de4355555557 => "333333333.33333343", 0xbecbf647612f3696 => "-0.0000033333333333333333",
            0x43143ff3c1cb0959 => "1424953923781206.2"]
            @test ecma_number(reinterpret(Float64, bits)) == s
        end
        # spec/plan.md §2.1's check values, and integers.
        for (x, s) in [(300.0, "300"), (0.1, "0.1"), (1e-6, "0.000001"), (1e-7, "1e-7"), (1.5e21, "1.5e+21"),
                       (-0.0, "0"), (300, "300"), (0, "0"), (-7, "-7"), (1e-9, "1e-9"), (600.5, "600.5")]
            @test ecma_number(x) == s
        end
        @test_throws ArgumentError ecma_number(NaN)
        @test_throws ArgumentError ecma_number(Inf)
        @test_throws ArgumentError ecma_number(2^60 + 1)
    end

    @testset "RFC 8785 strings and key order" begin
        # RFC 8785 §3.2.3's sorting example: UTF-16 code-unit order.
        keys_ = ["\u20ac", "\r", "\ufb33", "1", "\u0080", "\U0001f600", "\u00f6"]
        d = Dict(k => i for (i, k) in enumerate(keys_))
        @test jcs(d) == "{\"\\r\":2,\"1\":4,\"\u0080\":5,\"\u00f6\":7,\"\u20ac\":1,\"\U0001f600\":6,\"\ufb33\":3}"
        @test jcs(["\"\\\b\f\n\r\t\u0001\u001f\u007f\u2028"]) == "[\"\\\"\\\\\\b\\f\\n\\r\\t\\u0001\\u001f\u007f\u2028\"]"
        @test jcs(Dict("a" => nothing, "b" => true, "c" => false, "d" => Any[], "e" => Dict())) ==
              "{\"a\":null,\"b\":true,\"c\":false,\"d\":[],\"e\":{}}"
    end

    @testset "the hand-written expected plan is canonical" begin
        for (root, _, files) in walkdir(PFIX), f in files
            endswith(f, ".expected-plan.json") || continue
            text = read(joinpath(root, f), String)
            @test jcs(JSON.parse(text)) == text
        end
    end

    @testset "run-config schema" begin
        S = run_config_schema()
        valid = vcat(filter(endswith(".json"), readdir(joinpath(CONF, "run-configs", "valid"); join = true)),
            [joinpath(d, f) for d in readdir(PFIX; join = true) for f in readdir(d) if endswith(f, ".run.json")],
            [joinpath(d, "run.json") for d in readdir(joinpath(CONF, "plans", "refusals"); join = true) if isdir(d)])
        @test length(valid) > 40
        for f in valid
            @test schema_errors(S, JSON.parsefile(f)) === nothing
        end
        invalid = filter(f -> endswith(f, ".json") && !endswith(f, "expected-errors.json"),
            readdir(joinpath(CONF, "run-configs", "invalid"); join = true))
        @test length(invalid) == length(JSON.parsefile(joinpath(CONF, "run-configs", "invalid", "expected-errors.json")))
        for f in invalid
            @test schema_errors(S, JSON.parsefile(f)) !== nothing
        end
        # unevaluatedProperties on leaf nodes, and `$` anchoring at the very end.
        base = Dict("esm_integrator" => "0.1.0", "model" => "m.esm", "schedule" => Dict("leaf" => "a", "alg" => "Tsit5"))
        @test schema_errors(S, base) === nothing
        @test schema_errors(S, merge(base, Dict("schedule" => Dict("leaf" => "a", "bogus" => 1)))) !== nothing
        @test schema_errors(S, merge(base, Dict("schedule" => Dict("leaf" => "a\n")))) !== nothing
    end

    @testset "offsets (§5.2.3)" begin
        sizes = Dict("lon" => 8, "lat" => 3, "lev" => 4, "spc" => 5)
        ax = Axes(["lon", "lat"], "lev")
        shapes = Dict("q" => ["lon", "lat", "lev"], "c" => ["lon", "lev"], "s" => ["spc"], "w" => ["lon", "lat", "lev", "spc"])
        rows3 = [(1, 8), (1, 3), (1, 4)]
        idx3 = Union{String, Int}["i", "j", "k"]
        cls(bands...) = classify_bands(collect(BandEntry, bands), shapes, ax, sizes)
        band(col_idx; row = "q", col = "q", rows = rows3, row_idx = idx3, contracted = Tuple{String, Int, Int}[]) =
            BandEntry(row, col, rows, row_idx, Any[col_idx...], contracted)
        op(o, a, b) = Dict{String, Any}("op" => o, "args" => Any[a, b])
        # Scalar row and scalar column: no reach.
        @test cls(BandEntry("x", "y", Tuple{Int, Int}[], Union{String, Int}[], Any[])) == (:pointwise, nothing)
        # One scalar, one array: global, either way round.
        @test cls(BandEntry("x", "q", Tuple{Int, Int}[], Union{String, Int}[], Any["i", "j", "k"]))[1] === :global
        @test cls(BandEntry("q", "x", rows3, idx3, Any[]))[1] === :global
        # Both arrays. c equal to r: {0}.
        @test cls(band(["i", "j", "k"])) == (:pointwise, nothing)
        # r + δ, r − δ, δ + r on a horizontal axis: a stencil with that reach.
        @test cls(band([op("+", "i", 2), "j", "k"])) == (:stencil, Dict("lon" => (0, 2), "lat" => (0, 0)))
        @test cls(band([op("-", "i", 3), "j", "k"])) == (:stencil, Dict("lon" => (3, 0), "lat" => (0, 0)))
        @test cls(band(["i", op("+", 1, "j"), "k"])) == (:stencil, Dict("lon" => (0, 0), "lat" => (0, 1)))
        @test cls(band([op("+", "i", -1), "j", "k"])) == (:stencil, Dict("lon" => (1, 0), "lat" => (0, 0)))
        # The same on the vertical axis: column, offsets not reduced.
        @test cls(band(["i", "j", op("-", "k", 1)])) == (:column, nothing)
        @test raw_offsets(op("+", "k", 3), "k", 1, 4) == [3]
        # An integer constant against an integer row index: {k − ρ}.
        @test raw_offsets(7, 1, 1, 1) == [6]
        @test raw_offsets(op("*", 2, 3), 2, 2, 2) == [4]
        @test raw_offsets(op("-", op("+", 4, 4), 1), 8, 8, 8) == [-1]
        @test raw_offsets(Dict{String, Any}("op" => "-", "args" => Any[2]), 1, 1, 1) == [-3]
        @test raw_offsets(1, 1, 1, 1) == [0]
        # Integer constants (§5.2.3): n-ary + and *, one- or two-argument -, integer literals only,
        # exact in 64-bit integers.
        nop(o, xs...) = Dict{String, Any}("op" => o, "args" => Any[xs...])
        @test raw_offsets(nop("+", 1, 2, 3), 1, 1, 1) == [5]
        @test raw_offsets(nop("*", 2, 2, 2), 1, 1, 1) == [7]
        @test raw_offsets(nop("-", 1, 2, 3), 1, 1, 1) === nothing          # three-argument -
        @test raw_offsets(nop("+", "i", 1.0), "i", 1, 8) === nothing       # a float literal is not an integer
        @test raw_offsets(2.0, 1, 1, 1) === nothing
        @test raw_offsets(nop("*", typemax(Int), 2), 1, 1, 1) === nothing  # overflow: not a constant
        @test raw_offsets(nop("-", typemin(Int)), 1, 1, 1) === nothing
        @test raw_offsets(nop("+", nop("*", 2^31, 2^31), nop("-", 2^62)), 1, 1, 1) == [-1]
        # An integer constant against a name over [lo, hi]: {k − hi, …, k − lo}.
        @test raw_offsets(3, "i", 2, 4) == [-1, 0, 1]
        @test cls(band([3, "j", "k"]; rows = [(2, 4), (1, 3), (1, 4)])) == (:stencil, Dict("lon" => (1, 1), "lat" => (0, 0)))
        # Anything else covers the axis: another index name, a nonlinear index, a gather.
        @test raw_offsets("j", "i", 1, 8) === nothing
        @test raw_offsets(op("*", 2, "i"), "i", 1, 8) === nothing
        @test raw_offsets(op("+", "i", 1.0), "i", 1, 8) === nothing
        @test cls(band(["j", "j", "k"]))[1] === :global                      # horizontal covered: global
        @test cls(band(["i", "j", op("*", 2, "k")]))[1] === :column          # vertical covered: column
        @test cls(band(["i", "j", Dict{String, Any}("op" => "index", "args" => Any["tbl", "k"])]))[1] === :column
        # A column index referencing a contracted name covers its axis.
        kk = [("kk", 1, 4)]
        @test cls(band(["i", "j", "kk"]; contracted = kk)) == (:column, nothing)
        @test cls(band(["i", "j", op("+", "kk", 0)]; contracted = kk)) == (:column, nothing)
        @test cls(band(["kk", "j", "k"]; contracted = [("kk", 1, 8)]))[1] === :global
        # x has no dimension on X: the reach covers X (vertical: column; horizontal: global).
        @test cls(BandEntry("c", "q", [(1, 8), (1, 4)], Union{String, Int}["i", "k"], Any["i", "j", "k"]))[1] === :global
        @test cls(BandEntry("q", "c", rows3, idx3, Any["i", "k"])) == (:pointwise, nothing)
        shapes2 = merge(shapes, Dict("h" => ["lon", "lat"]))
        @test classify_bands([BandEntry("h", "q", [(1, 8), (1, 3)], Union{String, Int}["i", "j"], Any["i", "j", "kk"],
            [("kk", 1, 4)])], shapes2, ax, sizes) == (:column, nothing)
        @test classify_bands([BandEntry("h", "q", [(1, 8), (1, 3)], Union{String, Int}["i", "j"], Any["i", "j", 1])],
            shapes2, ax, sizes) == (:column, nothing)
        # An "other" axis (neither horizontal nor vertical) covered: global.
        @test cls(BandEntry("w", "s", [(1, 8), (1, 3), (1, 4), (1, 5)], Union{String, Int}["i", "j", "k", "m"], Any["n"]))[1] === :global
        @test cls(BandEntry("w", "s", [(1, 8), (1, 3), (1, 4), (1, 5)], Union{String, Int}["i", "j", "k", "m"], Any["m"])) ==
              (:pointwise, nothing)
    end

    @testset "periodic reduction (§5.2.3, D66)" begin
        # o' = ((o + ⌊N/2⌋) mod N) − ⌊N/2⌋, into [−⌊N/2⌋, ⌈N/2⌉ − 1].
        @test reduce_offset(7, 8) == -1               # the seam: cell 1 reads cell N
        @test reduce_offset(-7, 8) == 1
        @test reduce_offset(5, 8) == -3
        @test reduce_offset(4, 8) == -4 && reduce_offset(3, 8) == 3 && reduce_offset(-4, 8) == -4
        @test reduce_offset(63, 64) == -1
        @test [reduce_offset(o, 5) for o in -5:5] == [0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0]
        sizes = Dict("lon" => 8, "lat" => 2, "lev" => 2)
        shapes = Dict("q" => ["lon", "lat", "lev"])
        ax = Axes(["lon", "lat"], "lev")
        jk = ("j", "k")
        # periodic_stencil's 36 bands (expected.md): interior rows 4–6 read i−3 … i+2; the seam
        # rows read literal wrapped columns.
        bands = BandEntry[]
        for d in -3:2
            c = d == 0 ? "i" : Dict{String, Any}("op" => d < 0 ? "-" : "+", "args" => Any["i", abs(d)])
            push!(bands, BandEntry("q", "q", [(4, 6), (1, 2), (1, 2)], Union{String, Int}["i", jk...], Any[c, jk...]))
        end
        for r in (1, 2, 3, 7, 8), d in -3:2
            col = mod1(r + d, 8)
            push!(bands, BandEntry("q", "q", [(r, r), (1, 2), (1, 2)], Union{String, Int}[r, jk...], Any[col, jk...]))
        end
        @test length(bands) == 36
        @test classify_bands(bands, shapes, ax, sizes) == (:stencil, Dict("lon" => (3, 2), "lat" => (0, 0)))
        # Not reduced on the vertical axis: a wrapped vertical read is a column reach of N − 1.
        @test classify_bands([BandEntry("q", "q", [(1, 8), (1, 2), (1, 1)], Union{String, Int}["i", "j", 1], Any["i", "j", 2])],
            shapes, ax, sizes) == (:column, nothing)
    end

    @testset "locality of EarthSciASTDiff's own band lists" begin
        real = JSON.parsefile(joinpath(@__DIR__, "planner_bands.json"))
        bl(k) = BandEntry[BandEntry(d) for d in real[k]]
        @test classify_bands(bl("periodic_stencil"), Dict("Transport.q" => ["lon", "lat", "lev"]), Axes(["lon", "lat"], "lev"),
            Dict("lon" => 8, "lat" => 2, "lev" => 2)) == (:stencil, Dict("lon" => (3, 2), "lat" => (0, 0)))
        col = bl("column_contraction")
        @test col[2].contracted == [("kk_j", 1, 5)]
        @test classify_bands(col, Dict("Mix.q" => ["lon", "lat", "lev"]), Axes(["lon", "lat"], "lev"),
            Dict("lon" => 3, "lat" => 2, "lev" => 5)) == (:column, nothing)
        # Without axes, lev is horizontal and the contraction is global (refusal fixture auto-stiff-global).
        @test classify_bands(col, Dict("Mix.q" => ["lon", "lat", "lev"]), Axes(["lev", "lon"], nothing),
            Dict("lon" => 3, "lat" => 2, "lev" => 5))[1] === :global
        adv = bl("advection_reaction")
        shapes = Dict("Reaction.A" => ["cell"], "Reaction.B" => ["cell"])
        @test classify_bands(adv[2:5], shapes, Axes(["cell"], nothing), Dict("cell" => 64)) == (:stencil, Dict("cell" => (1, 0)))
        @test classify_bands(adv[[1, 6, 7, 8]], shapes, Axes(["cell"], nothing), Dict("cell" => 64)) == (:pointwise, nothing)
    end

    @testset "class and radius (§5.2.4)" begin
        sizes = Dict("lon" => 6, "lat" => 3)
        shapes = Dict("q" => ["lon", "lat"])
        ax = Axes(["lon", "lat"], nothing)
        b(ci, cj) = BandEntry("q", "q", [(1, 6), (1, 3)], Union{String, Int}["i", "j"], Any[ci, cj])
        op(o, a, n) = Dict{String, Any}("op" => o, "args" => Any[a, n])
        # A term with no bands is pointwise.
        @test classify_bands(BandEntry[], shapes, ax, sizes) == (:pointwise, nothing)
        @test classify_bands([b(op("-", "i", 2), "j"), b(op("+", "i", 2), "j")], shapes, ax, sizes) ==
              (:stencil, Dict("lon" => (2, 2), "lat" => (0, 0)))
        @test classify_bands([b(op("-", "i", 2), "j"), b(op("+", "i", 2), "j"), b("i", op("+", "j", 1)),
                              b("i", op("-", "j", 1))], shapes, ax, sizes) ==
              (:stencil, Dict("lon" => (2, 2), "lat" => (1, 1)))
        # The widest reach reduction allows, 3 + 2 + 1 = 6 on 6 cells, is still a stencil:
        # reduced offsets lie in [−⌊N/2⌋, ⌈N/2⌉ − 1], so lower + upper + 1 never exceeds N.
        @test classify_bands([b(op("+", "i", 2), "j"), b(op("-", "i", 3), "j")], shapes, ax, sizes) ==
              (:stencil, Dict("lon" => (3, 2), "lat" => (0, 0)))
        # Without a vertical axis, a model has no column terms.
        @test classify_bands([b("i", "j")], shapes, ax, sizes) == (:pointwise, nothing)
    end
end
