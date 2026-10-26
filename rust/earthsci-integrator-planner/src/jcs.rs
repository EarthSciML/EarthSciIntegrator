//! RFC 8785 JSON Canonicalization Scheme (spec/plan.md §2.1).
//!
//! Object keys are sorted by their UTF-16 code units, there is no
//! insignificant whitespace and no trailing newline, strings are escaped as
//! RFC 8785 §3.2.2.2 prescribes, and every number is written as ECMAScript's
//! `Number.prototype.toString` writes the binary64 value (RFC 8785 §3.2.2.3).
//! The writer sorts keys itself, so it does not depend on the order a
//! `serde_json::Map` keeps (EarthSciAST turns on `preserve_order`).

use serde_json::Value;
use std::fmt::Write as _;

/// The canonical form of `v`. Fails on a number that is not finite, which
/// JSON cannot carry.
pub fn to_string(v: &Value) -> Result<String, String> {
    let mut out = String::new();
    write_value(&mut out, v)?;
    Ok(out)
}

fn write_value(out: &mut String, v: &Value) -> Result<(), String> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            // JCS treats every number as a binary64 (RFC 8785 §3.2.2.3).
            let x = n
                .as_f64()
                .ok_or_else(|| format!("number {n} is not representable"))?;
            out.push_str(&format_number(x)?);
        }
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, &map[k])?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// A JSON string as RFC 8785 §3.2.2.2 writes it: `"` and `\` escaped, the
/// control characters U+0000–U+001F written `\b`, `\t`, `\n`, `\f`, `\r` or
/// `\u00xx` (lowercase hex), everything else as its UTF-8 bytes.
fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A binary64 value as ECMAScript's `Number.prototype.toString` writes it
/// (ECMA-262 `Number::toString`, radix 10): the shortest digit string that
/// round-trips, then fixed notation for decimal exponents in `[-6, 21)`
/// and exponential notation (`1e-7`, `1.5e+21`) outside it. −0 is `0`.
///
/// The digits come from Rust's shortest round-trip formatting (`{:e}`),
/// which, like ECMAScript, picks the digit string closest to the exact value
/// among the shortest ones.
pub fn format_number(x: f64) -> Result<String, String> {
    if !x.is_finite() {
        return Err(format!("{x} is not a JSON number"));
    }
    if x == 0.0 {
        return Ok("0".to_string());
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci
        .split_once('e')
        .ok_or_else(|| format!("unexpected float form {sci}"))?;
    let exp: i32 = exp
        .parse()
        .map_err(|_| format!("unexpected float form {sci}"))?;
    // The decimal digits s (k of them) and n, so that x = 0.s × 10^n.
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let mut n = exp + 1;
    if let Some((d, e)) = even_tie(x.abs(), &digits, exp) {
        digits = d;
        n = e + 1;
    }
    let k = digits.len() as i32;
    let mut out = String::new();
    if x < 0.0 {
        out.push('-');
    }
    if k <= n && n <= 21 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if n > 0 { '+' } else { '-' });
        let _ = write!(out, "{}", (n - 1).abs());
    }
    Ok(out)
}

/// ECMAScript's tie rule: when the exact value of `x` lies exactly halfway
/// between two `k`-digit candidates that both round-trip, the one with the
/// even last digit is chosen. Rust's shortest formatting may pick the other;
/// this returns the even candidate (digits, decimal exponent) in that case.
fn even_tie(x: f64, digits: &str, exp: i32) -> Option<(String, i32)> {
    let k = digits.len();
    // The exact decimal expansion of x (a binary64 has at most 767
    // significant decimal digits).
    let exact = format!("{:.800e}", x);
    let (m, e) = exact.split_once('e')?;
    let e: i32 = e.parse().ok()?;
    let all: String = m.chars().filter(|c| *c != '.').collect();
    if e != exp || all.len() <= k {
        return None;
    }
    let (prefix, rest) = all.split_at(k);
    let rest = rest.trim_end_matches('0');
    if rest != "5" {
        return None;
    }
    // Candidates: the truncated prefix and the prefix plus one unit.
    let lo = prefix.to_string();
    let mut hi: Vec<u8> = prefix.bytes().collect();
    let mut i = hi.len();
    let mut carry = true;
    while carry && i > 0 {
        i -= 1;
        if hi[i] == b'9' {
            hi[i] = b'0';
        } else {
            hi[i] += 1;
            carry = false;
        }
    }
    let (hi, hi_exp) = if carry {
        let mut s = String::from("1");
        s.push_str(&String::from_utf8(hi).ok()?[..k - 1]);
        (s, e + 1)
    } else {
        (String::from_utf8(hi).ok()?, e)
    };
    let parses = |d: &str, ex: i32| {
        format!("{d}e{}", ex - d.len() as i32 + 1)
            .parse::<f64>()
            .ok()
            == Some(x)
    };
    let even = |d: &str| (d.as_bytes()[d.len() - 1] - b'0').is_multiple_of(2);
    for (d, ex) in [(lo, e), (hi, hi_exp)] {
        if even(&d) && parses(&d, ex) {
            let d = d.trim_end_matches('0');
            let d = if d.is_empty() { "0" } else { d };
            return Some((d.to_string(), ex));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bits(b: u64) -> String {
        format_number(f64::from_bits(b)).unwrap()
    }

    /// RFC 8785 Appendix B.
    #[test]
    fn rfc8785_number_vectors() {
        let cases: &[(u64, &str)] = &[
            (0x0000000000000000, "0"),
            (0x8000000000000000, "0"),
            (0x0000000000000001, "5e-324"),
            (0x8000000000000001, "-5e-324"),
            (0x7fefffffffffffff, "1.7976931348623157e+308"),
            (0xffefffffffffffff, "-1.7976931348623157e+308"),
            (0x4340000000000000, "9007199254740992"),
            (0xc340000000000000, "-9007199254740992"),
            (0x4430000000000000, "295147905179352830000"),
            (0x44b52d02c7e14af5, "9.999999999999997e+22"),
            (0x44b52d02c7e14af6, "1e+23"),
            (0x44b52d02c7e14af7, "1.0000000000000001e+23"),
            (0x444b1ae4d6e2ef4e, "999999999999999700000"),
            (0x444b1ae4d6e2ef4f, "999999999999999900000"),
            (0x444b1ae4d6e2ef50, "1e+21"),
            (0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"),
            (0x3eb0c6f7a0b5ed8d, "0.000001"),
            (0x41b3de4355555553, "333333333.3333332"),
            (0x41b3de4355555554, "333333333.33333325"),
            (0x41b3de4355555555, "333333333.3333333"),
            (0x41b3de4355555556, "333333333.3333334"),
            (0x41b3de4355555557, "333333333.33333343"),
            (0xbecbf647612f3696, "-0.0000033333333333333333"),
            (0x43143ff3c1cb0959, "1424953923781206.2"),
        ];
        for (b, want) in cases {
            assert_eq!(bits(*b), *want, "bits {b:#018x}");
        }
        assert!(format_number(f64::NAN).is_err());
        assert!(format_number(f64::INFINITY).is_err());
    }

    /// spec/plan.md §2.1's check values.
    #[test]
    fn plan_check_values() {
        for (x, want) in [
            (300.0, "300"),
            (0.1, "0.1"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.5e21, "1.5e+21"),
            (-0.0, "0"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (123.456, "123.456"),
            (-1.5, "-1.5"),
            (5e-7, "5e-7"),
            (1.25e-5, "0.0000125"),
        ] {
            assert_eq!(format_number(x).unwrap(), want, "{x}");
        }
    }

    /// RFC 8785 §3.2.3's sorting example, and string escapes (§3.2.2.2).
    #[test]
    fn keys_and_strings() {
        let v = json!({
            "\u{20ac}": "Euro Sign",
            "\r": "Carriage Return",
            "\u{fb33}": "Hebrew Letter Dalet With Dagesh",
            "1": "One",
            "\u{1f600}": "Emoji: Grinning Face",
            "\u{80}": "Control",
            "\u{f6}": "Latin Small Letter O With Diaeresis"
        });
        let s = to_string(&v).unwrap();
        let order: Vec<&str> = vec![
            "\"\\r\"",
            "\"1\"",
            "\"\u{80}\"",
            "\"\u{f6}\"",
            "\"\u{20ac}\"",
            "\"\u{1f600}\"",
            "\"\u{fb33}\"",
        ];
        let mut last = 0;
        for k in order {
            let at = s.find(k).unwrap();
            assert!(at >= last, "{k} out of order in {s}");
            last = at;
        }
        let v = json!({"a": "\u{1}\u{1f}\"\\/\u{7f}\u{2028}\n\t"});
        assert_eq!(
            to_string(&v).unwrap(),
            "{\"a\":\"\\u0001\\u001f\\\"\\\\/\u{7f}\u{2028}\\n\\t\"}"
        );
        // RFC 8785 §3.2.2's example.
        let v = json!({
            "numbers": [333_333_333.333_333_3, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
            "string": "\u{20ac}$\u{000F}\nA'B\"\\\\\"/",
            "literals": [null, true, false]
        });
        assert_eq!(
            to_string(&v).unwrap(),
            "{\"literals\":[null,true,false],\"numbers\":[333333333.3333333,1e+30,4.5,0.002,1e-27],\"string\":\"\u{20ac}$\\u000f\\nA'B\\\"\\\\\\\\\\\"/\"}"
        );
    }
}
