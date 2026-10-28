# The repository paths the test files share. Each file that uses them includes
# this one when they are not yet defined, so the order they run in does not matter.

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CONF = joinpath(REPO, "tests", "conformance")
const PFIX = joinpath(CONF, "plans", "fixtures")
