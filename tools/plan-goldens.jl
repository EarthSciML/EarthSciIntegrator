# Writes the candidate planner goldens (spec/plan.md §9, tests/conformance/README.md):
# for every case `<case>.run.json` of every fixture under
# tests/conformance/plans/fixtures/, the Julia planner's plan directory
# `<case>.golden/` (plan.json, derived/*.esm, validation/*.bandvalues.json).
#
#     julia --project=julia/EarthSciIntegrator.jl tools/plan-goldens.jl [--out OUTDIR] [FIXTURE[/CASE] ...]
#
# Without OUTDIR the goldens are written beside the run configs, for review
# and check-in. With --out OUTDIR they go to OUTDIR/<fixture>/<case>.golden/. A
# case with `<case>.expected-plan.json` is compared with it, the derived
# documents' sha256 values aside (they are zeros there); a difference is
# printed. `write_plan` also writes `plan.report.json`, which carries package
# versions and measured errors and is not part of a golden; it is removed.
#
# The goldens are then reviewed by hand against each fixture's expected.md
# before they are checked in (spec/plan.md §9). Both test suites compare
# their planner's output with them, and a case without a golden fails.

using EarthSciIntegrator
using JSON

const FIXTURES = normpath(joinpath(@__DIR__, "..", "tests", "conformance", "plans", "fixtures"))

"""The hand-written expected plan with its zero hashes replaced by the golden's."""
function expected_with_hashes(expected::AbstractString, golden::AbstractDict)
    e = JSON.parse(expected)
    for (path, d) in e["derived"]
        haskey(golden["derived"], path) && (d["sha256"] = golden["derived"][path]["sha256"])
    end
    return jcs(e)
end

function main(args)
    outdir = nothing
    if !isempty(args) && startswith(args[1], "--out")
        a = popfirst!(args)
        outdir = a == "--out" ? (isempty(args) ? error("--out needs a directory") : popfirst!(args)) :
            startswith(a, "--out=") ? a[(length("--out=") + 1):end] : error("unknown option $a")
    end
    for a in args
        p = split(a, "/")
        ok = length(p) == 1 ? isdir(joinpath(FIXTURES, a)) :
            length(p) == 2 && isfile(joinpath(FIXTURES, p[1], "$(p[2]).run.json"))
        ok || error("$a names no fixture or case under $FIXTURES")
    end
    want = Set(args)
    failures = String[]
    for fixture in sort!(readdir(FIXTURES)), f in sort!(readdir(joinpath(FIXTURES, fixture)))
        endswith(f, ".run.json") || continue
        case = f[1:(end - length(".run.json"))]
        isempty(want) || fixture in want || "$fixture/$case" in want || continue
        dir = outdir === nothing ? joinpath(FIXTURES, fixture, "$case.golden") :
            joinpath(outdir, fixture, "$case.golden")
        print("$fixture/$case: ")
        t = @elapsed begin
            d = try
                plan(joinpath(FIXTURES, fixture, f))
            catch e
                e isa PlannerRefusal || rethrow()
                println("refused: ", sprint(showerror, e))
                push!(failures, "$fixture/$case")
                continue
            end
            write_plan(d, dir; bandvalues = true)
            rm(joinpath(dir, "plan.report.json"); force = true)
        end
        println("written to $dir ($(round(t; digits = 1)) s)")
        exp = joinpath(FIXTURES, fixture, "$case.expected-plan.json")
        if isfile(exp)
            got = read(joinpath(dir, "plan.json"), String)
            want_text = expected_with_hashes(read(exp, String), JSON.parse(got))
            if got == want_text
                println("  equals $case.expected-plan.json")
            else
                println("  DIFFERS from $case.expected-plan.json")
                push!(failures, "$fixture/$case (expected plan)")
            end
        end
    end
    isempty(failures) || (println("failures: ", join(failures, ", ")); exit(1))
    return
end

main(copy(ARGS))
