# The canonical JSON writer of plan.json (spec/plan.md §2.1): RFC 8785, the
# JSON Canonicalization Scheme. UTF-8, object keys sorted by UTF-16 code
# units, no insignificant whitespace, no trailing newline, numbers in
# ECMAScript's Number.prototype.toString form.

"""
    ecma_number(x::Real) -> String

`x` as ECMAScript's `Number.prototype.toString` writes it (ECMA-262
§6.1.6.1.20), which RFC 8785 §3.2.2.3 adopts: the shortest digit string that
round-trips, `-0` written `0`, plain decimal notation for decimal exponents in
`[-6, 21)`, exponential notation (`1e-7`, `1.5e+21`) otherwise. NaN and the
infinities have no JSON form and are an `ArgumentError`.

Integers are written as the double they denote (RFC 8785 treats every JSON
number as an IEEE 754 double); one that a double cannot hold exactly is an
`ArgumentError` rather than a silently rounded value.
"""
function ecma_number(x::Integer)
    x isa Bool && throw(ArgumentError("a Bool is not a JSON number"))
    f = Float64(x)
    (isfinite(f) && f == x) || throw(ArgumentError("integer $x has no exact binary64 form"))
    return ecma_number(f)
end

function ecma_number(x::AbstractFloat)
    x = Float64(x)
    isfinite(x) || throw(ArgumentError("$x has no JSON form (RFC 8785 §3.2.2.3)"))
    x == 0 && return "0"                      # +0 and -0
    neg = x < 0
    # Julia's `repr` of a Float64 is the shortest round-trip digit string (Ryu),
    # the same digits ECMA-262 asks for: the fewest digits `k`, and among
    # those the value closest to `x`. Only the layout differs.
    s = repr(abs(x))
    mant, e10 = occursin('e', s) ? (split(s, 'e')[1], parse(Int, split(s, 'e')[2])) : (s, 0)
    ip, fp = occursin('.', mant) ? (split(mant, '.')[1], split(mant, '.')[2]) : (mant, "")
    digits = ip * fp
    point = length(ip) + e10                  # value = 0.digits × 10^point
    nlead = something(findfirst(!=('0'), digits), length(digits) + 1) - 1
    digits = digits[nlead+1:end]
    point -= nlead
    digits = rstrip(digits, '0')
    k, n = length(digits), point              # ECMA-262: s × 10^(n−k), k digits
    out = if k <= n <= 21
        digits * "0"^(n - k)
    elseif 0 < n <= 21
        digits[1:n] * "." * digits[n+1:end]
    elseif -6 < n <= 0
        "0." * "0"^(-n) * digits
    else
        e = n - 1
        es = e < 0 ? "-" * string(-e) : "+" * string(e)
        k == 1 ? digits * "e" * es : digits[1:1] * "." * digits[2:end] * "e" * es
    end
    return neg ? "-" * out : out
end

"""
    jcs_string_literal(io, s)

A JSON string as RFC 8785 §3.2.2.2 writes it: `"` and `\\` escaped, the
control characters below U+0020 as `\\b`, `\\t`, `\\n`, `\\f`, `\\r` or
`\\u00xx` (lowercase hex), everything else as its UTF-8 bytes.
"""
function jcs_string_literal(io::IO, s::AbstractString)
    isvalid(s) || throw(ArgumentError("string is not valid UTF-8: $(repr(s))"))
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c < ' '
            if c == '\b'
                print(io, "\\b")
            elseif c == '\t'
                print(io, "\\t")
            elseif c == '\n'
                print(io, "\\n")
            elseif c == '\f'
                print(io, "\\f")
            elseif c == '\r'
                print(io, "\\r")
            else
                print(io, "\\u", string(UInt32(c); base = 16, pad = 4))
            end
        else
            print(io, c)
        end
    end
    print(io, '"')
end

"""
    utf16_isless(a, b)

Order of object keys under RFC 8785 §3.2.3: lexicographic over UTF-16 code
units (which differs from code-point order only for characters above U+FFFF
against U+E000–U+FFFF).
"""
utf16_isless(a::AbstractString, b::AbstractString) = transcode(UInt16, String(a)) < transcode(UInt16, String(b))

jcs_write(io::IO, ::Nothing) = print(io, "null")
jcs_write(io::IO, x::Bool) = print(io, x ? "true" : "false")
jcs_write(io::IO, x::Real) = print(io, ecma_number(x))
jcs_write(io::IO, s::AbstractString) = jcs_string_literal(io, s)
jcs_write(io::IO, s::Symbol) = jcs_string_literal(io, String(s))

function jcs_write(io::IO, v::Union{AbstractVector, Tuple})
    print(io, '[')
    first = true
    for x in v
        first || print(io, ',')
        first = false
        jcs_write(io, x)
    end
    print(io, ']')
end

function jcs_write(io::IO, d::AbstractDict)
    ks = [String(k) for k in keys(d)]
    allunique(ks) || throw(ArgumentError("object has duplicate keys"))
    sort!(ks; lt = utf16_isless)
    print(io, '{')
    for (i, k) in enumerate(ks)
        i > 1 && print(io, ',')
        jcs_string_literal(io, k)
        print(io, ':')
        jcs_write(io, haskey(d, k) ? d[k] : d[Symbol(k)])
    end
    print(io, '}')
end

"""
    jcs(x) -> String

`x` (dictionaries, vectors, strings, numbers, booleans, `nothing`) serialized
as RFC 8785 canonical JSON (spec/plan.md §2.1).
"""
jcs(x) = sprint(jcs_write, x)
