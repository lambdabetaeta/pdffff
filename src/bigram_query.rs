// SPDX-License-Identifier: MIT
// Adapted from fff's `bigram_query.rs`, (c) 2025 Dmitriy Kovalenko, MIT.
// Upstream: <https://github.com/dmitriy-kovalenko/fff>.
//
// Modifications for pdffff:
//   * The query evaluates against pdffff's `BigramIndex` (see
//     `crate::bigram`) rather than fff's `BigramFilter`. The internal
//     layout is the same (a `lookup: Vec<u16>` plus a flat `dense_data`
//     slab at fixed stride `words`), so the evaluator code is a
//     line-for-line port.
//   * fff's `SmallVec<[u8; 16]>` is replaced with `Vec<u8>` capped at
//     `MAX_CLASS_EXPAND = 16` elements. The cap is enforced by every
//     producer and the data structure is logically the same — small,
//     deduped, capped — but uses one fewer transitive dependency
//     (`smallvec`).
//   * fff's literal-byte temporary `SmallVec<[u8; 64]>` is also a
//     plain `Vec<u8>` for the same reason.
//   * Public visibility on the two driver entry points
//     (`regex_to_bigram_query`, `fuzzy_to_bigram_query`) is widened
//     from `pub(crate)` (fff) to `pub` so `src/query.rs` can call them
//     across the module boundary.
//   * The test module's `build_test_index` calls our
//     `BigramIndexBuilder::compress(Some(0))` so that single-document
//     bigrams survive the density filter even on a tiny corpus.

//! Regex → bigram decomposition for the inverted bigram index.
//!
//! Parses a regex pattern with `regex-syntax`, walks the HIR to extract
//! guaranteed bigram keys (u16), and evaluates them as an AND/OR query
//! tree against [`crate::bigram::BigramIndex`]'s inverted posting lists.
//!
//! Two bigram types are extracted:
//! - **Consecutive** (gap=0): adjacent byte pairs `(pattern[i], pattern[i+1])`
//! - **Sparse-1** (gap=1): pairs across a single-byte wildcard, e.g.
//!   `a.b → (a,b)`
//!
//! The sparse-1 extraction is the key feature: regex patterns like
//! `foo.bar` yield the cross-boundary sparse-1 bigram `(o,b)` that
//! provides strong filtering even when the `.` prevents any consecutive
//! cross-boundary bigram.

use crate::bigram::BigramIndex;
use regex_syntax::hir::{Class, Hir, HirKind};
use std::borrow::Cow;

/// Maximum byte values to enumerate from a character class.
/// Larger classes are treated as unknown (no bigram extractable).
const MAX_CLASS_EXPAND: usize = 16;

/// Sentinel value matching fff's "no column" marker for the bigram
/// lookup table.
const NO_COLUMN: u16 = u16::MAX;

#[inline]
fn consec_key(a: u8, b: u8) -> Option<u16> {
    let al = a.to_ascii_lowercase();
    let bl = b.to_ascii_lowercase();
    if (32..=126).contains(&al) && (32..=126).contains(&bl) {
        Some((al as u16) << 8 | bl as u16)
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub enum BigramQuery {
    Any,
    /// A consecutive bigram key to look up in the main index.
    Consec(u16),
    /// A skip-1 bigram key to look up in the skip sub-index.
    Skip1(u16),
    /// All children must match (intersect posting lists).
    And(Vec<BigramQuery>),
    /// At least one child must match (union posting lists).
    Or(Vec<BigramQuery>),
}

/// SIMD-friendly bitwise OR of two equal-length bitsets.
#[inline]
fn bitset_or(a: &mut [u64], b: &[u64]) {
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x |= *y);
}

/// SIMD-friendly bitwise AND of two equal-length bitsets.
#[inline]
fn bitset_and(a: &mut [u64], b: &[u64]) {
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x &= *y);
}

impl BigramQuery {
    pub fn is_any(&self) -> bool {
        matches!(self, BigramQuery::Any)
    }

    /// Evaluate the query against `index`, returning a candidate bitset
    /// or `None` if the query has no filterable information.
    pub fn evaluate(&self, index: &BigramIndex) -> Option<Vec<u64>> {
        self.evaluate_cow(index).map(Cow::into_owned)
    }

    fn evaluate_cow<'a>(&self, index: &'a BigramIndex) -> Option<Cow<'a, [u64]>> {
        match self {
            BigramQuery::Any => None,

            BigramQuery::Consec(key) => {
                let col = index.lookup()[*key as usize];
                if col == NO_COLUMN {
                    return None;
                }
                let words = index.words();
                let offset = col as usize * words;
                let data = index.dense_data();
                if offset + words > data.len() {
                    return None;
                }
                Some(Cow::Borrowed(&data[offset..offset + words]))
            }

            BigramQuery::Skip1(key) => {
                let skip = index.skip_index()?;
                let col = skip.lookup()[*key as usize];
                if col == NO_COLUMN {
                    return None;
                }
                let words = skip.words();
                let offset = col as usize * words;
                let data = skip.dense_data();
                if offset + words > data.len() {
                    return None;
                }
                Some(Cow::Borrowed(&data[offset..offset + words]))
            }

            BigramQuery::And(children) => {
                let mut result: Option<Vec<u64>> = None;
                for child in children {
                    if let Some(child_bits) = child.evaluate_cow(index) {
                        result = Some(match result {
                            None => child_bits.into_owned(),
                            Some(mut r) => {
                                bitset_and(&mut r, &child_bits);
                                r
                            }
                        });
                    }
                }
                result.map(Cow::Owned)
            }

            BigramQuery::Or(children) => {
                if children.is_empty() {
                    return None;
                }
                let mut result: Option<Vec<u64>> = None;
                for child in children {
                    match child.evaluate_cow(index) {
                        // Any branch can't be filtered → whole OR can't be filtered
                        None => return None,
                        Some(child_bits) => {
                            result = Some(match result {
                                None => child_bits.into_owned(),
                                Some(mut r) => {
                                    bitset_or(&mut r, &child_bits);
                                    r
                                }
                            });
                        }
                    }
                }
                result.map(Cow::Owned)
            }
        }
    }
}

/// Intermediate state tracked during HIR traversal for bigram extraction.
struct HirInfo {
    query: BigramQuery,
    /// Possible first bytes (lowercased, printable ASCII) when this node matches.
    first: Option<Vec<u8>>,
    /// Possible last bytes.
    last: Option<Vec<u8>>,
    /// Whether this node can match the empty string.
    can_be_empty: bool,
}

impl HirInfo {
    fn empty() -> Self {
        Self {
            query: BigramQuery::Any,
            first: None,
            last: None,
            can_be_empty: true,
        }
    }
}

/// Prefilter fuzzy query. The algorithm is the following:
/// we allow max_typos = min(len/3,2). Every typo destroys at most 2
/// consecutive bigrams, so out of N bigrams at least N - 2 * max_typos
/// have to be present in the matching file.
pub fn fuzzy_to_bigram_query(query: &str, num_probes: usize) -> BigramQuery {
    let lower: Vec<u8> = query.bytes().map(|b| b.to_ascii_lowercase()).collect();

    if lower.len() < 2 {
        return BigramQuery::Any;
    }

    let max_typos = (lower.len() / 3).min(2);

    // Extract all consecutive bigram keys.
    let bigram_keys: Vec<u16> = lower
        .windows(2)
        .filter_map(|w| consec_key(w[0], w[1]))
        .collect();

    if bigram_keys.is_empty() {
        return BigramQuery::Any;
    }

    // For very short queries (0 typos), AND all bigrams — exact subsequence.
    if max_typos == 0 {
        return simplify_and(
            bigram_keys
                .iter()
                .map(|&k| BigramQuery::Consec(k))
                .collect(),
        );
    }

    // Pick evenly-spaced probe bigrams.
    let n = num_probes.min(bigram_keys.len());
    if n <= max_typos {
        // Too few probes to require anything useful.
        return simplify_or(
            bigram_keys
                .iter()
                .map(|&k| BigramQuery::Consec(k))
                .collect(),
        );
    }

    let probes: Vec<u16> = if n == bigram_keys.len() {
        bigram_keys
    } else {
        (0..n)
            .map(|i| {
                let idx = i * (bigram_keys.len() - 1) / (n - 1);
                bigram_keys[idx]
            })
            .collect()
    };

    let required = n - max_typos;

    // If required == n, just AND all probes.
    if required >= n {
        return simplify_and(probes.iter().map(|&k| BigramQuery::Consec(k)).collect());
    }

    // Generate all C(n, required) subsets → OR(AND(subset), ...)
    let mut branches = Vec::new();
    let mut combo = vec![0u16; required];
    combine(&probes, required, 0, 0, &mut combo, &mut branches);

    simplify_or(branches)
}

/// Build C(n, k) combination branches in-place on a fixed-size slice.
fn combine(
    items: &[u16],
    k: usize,
    start: usize,
    depth: usize,
    combo: &mut [u16],
    branches: &mut Vec<BigramQuery>,
) {
    if depth == k {
        branches.push(simplify_and(
            combo.iter().map(|&key| BigramQuery::Consec(key)).collect(),
        ));
        return;
    }
    let remaining = k - depth;
    for i in start..=items.len() - remaining {
        combo[depth] = items[i];
        combine(items, k, i + 1, depth + 1, combo, branches);
    }
}

pub fn regex_to_bigram_query(pattern: &str) -> BigramQuery {
    let mut parser = regex_syntax::ParserBuilder::new()
        .unicode(false)
        .utf8(false)
        .build();

    let hir = match parser.parse(pattern) {
        Ok(h) => h,
        Err(_) => return BigramQuery::Any,
    };

    decompose(&hir).query
}

fn decompose(hir: &Hir) -> HirInfo {
    let can_be_empty = hir.properties().minimum_len().is_none_or(|n| n == 0);

    match hir.kind() {
        HirKind::Empty => HirInfo::empty(),

        HirKind::Literal(lit) => decompose_literal(lit.0.as_ref()),

        HirKind::Class(class) => {
            let bytes = expand_class(class);
            match bytes {
                Some(b) if !b.is_empty() => HirInfo {
                    query: BigramQuery::Any,
                    first: Some(b.clone()),
                    last: Some(b),
                    can_be_empty,
                },
                _ => HirInfo {
                    query: BigramQuery::Any,
                    first: None,
                    last: None,
                    can_be_empty,
                },
            }
        }

        HirKind::Look(_) => HirInfo::empty(),

        HirKind::Repetition(rep) => {
            let inner = decompose(&rep.sub);
            if rep.min == 0 {
                HirInfo {
                    query: BigramQuery::Any,
                    first: inner.first,
                    last: inner.last,
                    can_be_empty: true,
                }
            } else {
                // min >= 1: inner bigrams guaranteed
                let mut qs = Vec::new();
                if !inner.query.is_any() {
                    qs.push(inner.query.clone());
                }
                // min >= 2: cross-boundary between consecutive occurrences
                if rep.min >= 2 {
                    push_cross_consec(&mut qs, inner.last.as_deref(), inner.first.as_deref());
                }
                HirInfo {
                    query: simplify_and(qs),
                    first: inner.first,
                    last: inner.last,
                    can_be_empty,
                }
            }
        }

        HirKind::Capture(cap) => decompose(&cap.sub),

        HirKind::Concat(parts) => decompose_concat(parts),

        HirKind::Alternation(alts) => decompose_alternation(alts),
    }
}

/// Extract bigrams from a literal byte sequence.
fn decompose_literal(bytes: &[u8]) -> HirInfo {
    if bytes.is_empty() {
        return HirInfo::empty();
    }

    let lower: Vec<u8> = bytes.iter().map(|b| b.to_ascii_lowercase()).collect();

    if lower.len() == 1 {
        let b = lower[0];
        let first = if (32..=126).contains(&b) {
            Some(vec![b])
        } else {
            None
        };
        return HirInfo {
            query: BigramQuery::Any,
            first: first.clone(),
            last: first,
            can_be_empty: false,
        };
    }

    let mut qs: Vec<BigramQuery> = Vec::new();

    // Consecutive bigrams
    for w in lower.windows(2) {
        if let Some(k) = consec_key(w[0], w[1]) {
            qs.push(BigramQuery::Consec(k));
        }
    }

    // Skip-1 bigrams from the literal itself
    if lower.len() >= 3 {
        for i in 0..lower.len() - 2 {
            if let Some(k) = consec_key(lower[i], lower[i + 2]) {
                qs.push(BigramQuery::Skip1(k));
            }
        }
    }

    let first_byte = lower[0];
    let last_byte = *lower.last().unwrap();

    HirInfo {
        query: simplify_and(qs),
        first: if (32..=126).contains(&first_byte) {
            Some(vec![first_byte])
        } else {
            None
        },
        last: if (32..=126).contains(&last_byte) {
            Some(vec![last_byte])
        } else {
            None
        },
        can_be_empty: false,
    }
}

fn decompose_concat(parts: &[Hir]) -> HirInfo {
    if parts.is_empty() {
        return HirInfo::empty();
    }

    let infos: Vec<HirInfo> = parts.iter().map(decompose).collect();
    let mut qs: Vec<BigramQuery> = Vec::new();

    // 1. Collect child bigrams
    for info in &infos {
        if !info.query.is_any() {
            qs.push(info.query.clone());
        }
    }

    // 2. Dense cross-boundary between adjacent mandatory parts
    for pair in infos.windows(2) {
        if !pair[0].can_be_empty && !pair[1].can_be_empty {
            push_cross_consec(&mut qs, pair[0].last.as_deref(), pair[1].first.as_deref());
        }
    }

    // 3. Sparse-1 cross-boundary: across a single 1-byte-wide middle part.
    //    Catches `foo.bar` → sparse-1 `(o,b)` across the dot.
    if parts.len() >= 3 {
        for i in 0..parts.len() - 2 {
            let left = &infos[i];
            let mid = &parts[i + 1];
            let right = &infos[i + 2];

            let min_len = mid.properties().minimum_len();
            let max_len = mid.properties().maximum_len();
            let is_1byte = min_len == Some(1) && max_len == Some(1);

            if is_1byte && !left.can_be_empty && !right.can_be_empty {
                push_cross_skip1(&mut qs, left.last.as_deref(), right.first.as_deref());
            }
        }
    }

    let first = collect_first(&infos);
    let last = collect_last(&infos);
    let can_be_empty = infos.iter().all(|i| i.can_be_empty);

    HirInfo {
        query: simplify_and(qs),
        first,
        last,
        can_be_empty,
    }
}

fn decompose_alternation(alts: &[Hir]) -> HirInfo {
    if alts.is_empty() {
        return HirInfo::empty();
    }

    let infos: Vec<HirInfo> = alts.iter().map(decompose).collect();
    let query = simplify_or(infos.iter().map(|i| i.query.clone()).collect());
    let first = merge_byte_sets(infos.iter().map(|i| &i.first));
    let last = merge_byte_sets(infos.iter().map(|i| &i.last));
    let can_be_empty = infos.iter().any(|i| i.can_be_empty);

    HirInfo {
        query,
        first,
        last,
        can_be_empty,
    }
}

fn expand_class(class: &Class) -> Option<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    match class {
        Class::Bytes(bc) => {
            for range in bc.ranges() {
                let count = (range.end() as usize) - (range.start() as usize) + 1;
                if bytes.len() + count > MAX_CLASS_EXPAND {
                    return None;
                }
                for b in range.start()..=range.end() {
                    if (32..=126).contains(&b) {
                        let lower = b.to_ascii_lowercase();
                        if !bytes.contains(&lower) {
                            bytes.push(lower);
                        }
                    }
                }
            }
        }
        Class::Unicode(uc) => {
            for range in uc.ranges() {
                let start = range.start() as u32;
                let end = range.end() as u32;
                if start > 127 {
                    continue;
                }
                let ascii_end = end.min(126) as u8;
                let ascii_start = start.max(32) as u8;
                if ascii_start > ascii_end {
                    continue;
                }
                let count = (ascii_end - ascii_start) as usize + 1;
                if bytes.len() + count > MAX_CLASS_EXPAND {
                    return None;
                }
                for b in ascii_start..=ascii_end {
                    let lower = b.to_ascii_lowercase();
                    if !bytes.contains(&lower) {
                        bytes.push(lower);
                    }
                }
            }
        }
    }
    if bytes.is_empty() { None } else { Some(bytes) }
}

/// Push consecutive cross-product bigrams into `qs`.
fn push_cross_consec(qs: &mut Vec<BigramQuery>, last: Option<&[u8]>, first: Option<&[u8]>) {
    if let Some(q) = cross_product(last, first, false) {
        qs.push(q);
    }
}

/// Push skip-1 cross-product bigrams into `qs`.
fn push_cross_skip1(qs: &mut Vec<BigramQuery>, last: Option<&[u8]>, first: Option<&[u8]>) {
    if let Some(q) = cross_product(last, first, true) {
        qs.push(q);
    }
}

fn cross_product(last: Option<&[u8]>, first: Option<&[u8]>, skip: bool) -> Option<BigramQuery> {
    let last = last?;
    let first = first?;
    let n = last.len() * first.len();
    if n == 0 || n > MAX_CLASS_EXPAND * MAX_CLASS_EXPAND {
        return None;
    }

    let mut bigrams: Vec<BigramQuery> = Vec::with_capacity(n);
    for &l in last {
        for &f in first {
            if let Some(k) = consec_key(l, f) {
                let node = if skip {
                    BigramQuery::Skip1(k)
                } else {
                    BigramQuery::Consec(k)
                };
                bigrams.push(node);
            }
        }
    }

    match bigrams.len() {
        0 => None,
        1 => Some(bigrams.into_iter().next().unwrap()),
        _ => Some(simplify_or(bigrams)),
    }
}

fn collect_first(infos: &[HirInfo]) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = Vec::new();
    for info in infos {
        if let Some(ref bytes) = info.first {
            for &b in bytes {
                if !result.contains(&b) {
                    if result.len() >= MAX_CLASS_EXPAND {
                        return None;
                    }
                    result.push(b);
                }
            }
        } else if !info.can_be_empty {
            return None;
        }
        if !info.can_be_empty {
            break;
        }
    }
    if result.is_empty() { None } else { Some(result) }
}

fn collect_last(infos: &[HirInfo]) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = Vec::new();
    for info in infos.iter().rev() {
        if let Some(ref bytes) = info.last {
            for &b in bytes {
                if !result.contains(&b) {
                    if result.len() >= MAX_CLASS_EXPAND {
                        return None;
                    }
                    result.push(b);
                }
            }
        } else if !info.can_be_empty {
            return None;
        }
        if !info.can_be_empty {
            break;
        }
    }
    if result.is_empty() { None } else { Some(result) }
}

fn merge_byte_sets<'a>(
    iter: impl Iterator<Item = &'a Option<Vec<u8>>>,
) -> Option<Vec<u8>> {
    let mut result: Vec<u8> = Vec::new();
    for opt in iter {
        match opt {
            None => return None,
            Some(bytes) => {
                for &b in bytes {
                    if !result.contains(&b) {
                        if result.len() >= MAX_CLASS_EXPAND {
                            return None;
                        }
                        result.push(b);
                    }
                }
            }
        }
    }
    if result.is_empty() { None } else { Some(result) }
}

fn simplify_and(children: Vec<BigramQuery>) -> BigramQuery {
    let mut flat: Vec<BigramQuery> = Vec::new();
    for child in children {
        match child {
            BigramQuery::Any => {}
            BigramQuery::And(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    match flat.len() {
        0 => BigramQuery::Any,
        1 => flat.into_iter().next().unwrap(),
        _ => BigramQuery::And(flat),
    }
}

fn simplify_or(children: Vec<BigramQuery>) -> BigramQuery {
    if children.iter().any(|c| c.is_any()) {
        return BigramQuery::Any;
    }
    let mut flat: Vec<BigramQuery> = Vec::new();
    for child in children {
        match child {
            BigramQuery::Or(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    match flat.len() {
        0 => BigramQuery::Any,
        1 => flat.into_iter().next().unwrap(),
        _ => BigramQuery::Or(flat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigram::BigramIndexBuilder;

    /// Build a tiny index from the given file contents for testing.
    ///
    /// We pass `Some(0)` to `compress` so single-document bigrams survive
    /// the density filter on tiny corpora — without this every column
    /// would be dropped because pdffff's default heuristic targets a
    /// chunk corpus, not the 3-row toy corpora here.
    fn build_test_index(files: &[&[u8]]) -> BigramIndex {
        let n = files.len();
        let consec_builder = BigramIndexBuilder::new(n);
        let skip_builder = BigramIndexBuilder::new(n);
        for (i, content) in files.iter().enumerate() {
            consec_builder.add_item_content(&skip_builder, i, content);
        }
        let mut idx = consec_builder.compress(Some(0));
        idx.set_skip_index(skip_builder.compress(Some(0)));
        idx
    }

    #[test]
    fn literal_pattern() {
        let idx = build_test_index(&[
            b"hello world",     // 0: contains "hello"
            b"goodbye world",   // 1: no "hello"
            b"say hello there", // 2: contains "hello"
        ]);

        let q = regex_to_bigram_query("hello");
        assert!(!q.is_any());

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        assert!(!BigramIndex::is_candidate(&candidates, 1));
        assert!(BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn alternation() {
        let idx = build_test_index(&[
            b"has foo in it", // 0
            b"has bar in it", // 1
            b"has xyz in it", // 2
        ]);

        let q = regex_to_bigram_query("foo|bar");
        assert!(!q.is_any());

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        assert!(BigramIndex::is_candidate(&candidates, 1));
        // xyz doesn't contain foo or bar bigrams
        assert!(!BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn wildcard_concat() {
        let idx = build_test_index(&[
            b"foo something bar", // 0
            b"foo only",          // 1: has foo but not bar
            b"only bar",          // 2: has bar but not foo
        ]);

        let q = regex_to_bigram_query("foo.*bar");
        assert!(!q.is_any());

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        // file 1 and 2 should be filtered (missing bigrams from "bar" / "foo")
        assert!(!BigramIndex::is_candidate(&candidates, 1));
        assert!(!BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn sparse1_across_dot() {
        // "a.b" should produce a skip-1 bigram (a,b)
        let idx = build_test_index(&[
            b"axb", // 0: has sparse-1 (a,b)
            b"ayb", // 1: has sparse-1 (a,b)
            b"xyz", // 2: no (a,b) at all
        ]);

        let q = regex_to_bigram_query("a.b");
        assert!(!q.is_any());

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        assert!(BigramIndex::is_candidate(&candidates, 1));
        assert!(!BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn sparse1_across_digit() {
        // "foo\dbar" → sparse-1 (o,b) across \d
        let idx = build_test_index(&[
            b"foo3bar baz", // 0: has all bigrams
            b"foobar baz",  // 1: has consecutive (o,b) but pattern needs sparse-1
            b"xyz only",    // 2: no relevant bigrams
        ]);

        let q = regex_to_bigram_query(r"foo\dbar");
        assert!(!q.is_any());

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        // file 1 may or may not match depending on what bigrams are in the index
        // The important thing is file 2 is excluded:
        assert!(!BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn pure_wildcard_is_any() {
        let q = regex_to_bigram_query(".*");
        assert!(q.is_any());
    }

    #[test]
    fn single_char_is_any() {
        let q = regex_to_bigram_query("a");
        assert!(q.is_any());
    }

    #[test]
    fn invalid_regex_is_any() {
        let q = regex_to_bigram_query("[invalid");
        assert!(q.is_any());
    }

    #[test]
    fn optional_group_excluded() {
        // (bar)? is optional — its bigrams are not required
        let q = regex_to_bigram_query("foo(bar)?baz");
        assert!(!q.is_any());

        let idx = build_test_index(&[
            b"foobaz content",    // 0: has foo+baz bigrams (bar absent)
            b"foobarbaz content", // 1: has everything
            b"xyz only",          // 2: nothing
        ]);

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        assert!(BigramIndex::is_candidate(&candidates, 1));
        assert!(!BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn repetition_min2_cross_boundary() {
        // (ab){2,} → bigram "ab" + cross-boundary "b","a"
        let q = regex_to_bigram_query("(ab){2,}");
        assert!(!q.is_any());

        let idx = build_test_index(&[
            b"ababab", // 0: has "ab" and "b"->"a"
            b"abonly", // 1: has "ab" but not "b"->"a"
            b"xyz",    // 2: nothing
        ]);

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        assert!(!BigramIndex::is_candidate(&candidates, 2));
    }

    #[test]
    fn two_dots_no_sparse1() {
        // "a..b" — two 1-byte parts between a and b, not a single 1-byte part
        // No sparse-1 (a,b) should be extracted
        let q = regex_to_bigram_query("a..b");
        // Single-char literals with 2 unknown bytes between → Any
        assert!(q.is_any());
    }

    #[test]
    fn character_class_cross_boundary() {
        // [abc]de → cross-boundary OR(ad,bd,cd) + bigram de
        // All three class variants must appear in the corpus so the OR
        // branches are tracked in the index.
        let idx = build_test_index(&[
            b"ade content", // 0: has ad
            b"bde content", // 1: has bd
            b"cde content", // 2: has cd
            b"xde content", // 3: has de but not ad/bd/cd
        ]);

        let q = regex_to_bigram_query("[abc]de");
        assert!(!q.is_any());

        let candidates = q.evaluate(&idx).unwrap();
        assert!(BigramIndex::is_candidate(&candidates, 0));
        assert!(BigramIndex::is_candidate(&candidates, 1));
        assert!(BigramIndex::is_candidate(&candidates, 2));
        // file 3 doesn't have ad/bd/cd so should be filtered
        assert!(!BigramIndex::is_candidate(&candidates, 3));
    }

    // ── Helpers for inspecting query trees ──────────────────────────

    fn has_consec(q: &BigramQuery, a: u8, b: u8) -> bool {
        let Some(key) = consec_key(a, b) else {
            return false;
        };
        match q {
            BigramQuery::Consec(k) => *k == key,
            BigramQuery::And(cs) | BigramQuery::Or(cs) => cs.iter().any(|c| has_consec(c, a, b)),
            _ => false,
        }
    }

    fn has_skip1(q: &BigramQuery, a: u8, b: u8) -> bool {
        let Some(key) = consec_key(a, b) else {
            return false;
        };
        match q {
            BigramQuery::Skip1(k) => *k == key,
            BigramQuery::And(cs) | BigramQuery::Or(cs) => cs.iter().any(|c| has_skip1(c, a, b)),
            _ => false,
        }
    }

    /// Bigram expectation: `("ab", is_skip1)`.
    /// The 2-char str is the byte pair; C = consecutive, S = skip-1.
    type Bg = (&'static str, bool);
    const C: bool = false;
    const S: bool = true;

    /// Top 15+ commonly used regex patterns from
    /// <https://digitalfortress.tech/tips/top-15-commonly-used-regex/>
    /// plus typical grep patterns used by agentic tools.
    ///
    /// Each entry: `(regex, Option<&[Bg]>)`.
    ///   - `None`        → pure classes / unsupported syntax, Any is acceptable.
    ///   - `Some(&[..])` → must be non-Any, and every listed bigram must appear.
    #[test]
    fn common_regex_patterns() {
        #[rustfmt::skip]
        let cases: &[(&str, Option<&[Bg]>)] = &[
            // ── Pure-class / anchor / unsupported → Any is fine ──────
            (r"^\d+$",                                                      None), // 1.  whole numbers
            (r"^\d*\.\d+$",                                                 None), // 2.  decimals
            (r"^[a-zA-Z0-9]*$",                                             None), // 3.  alphanumeric
            (r"^[a-zA-Z0-9 ]*$",                                            None), // 4.  alphanum + space
            (r"^[a-z0-9_-]{3,16}$",                                         None), // 5.  username
            (r"[12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])",           None), // 6.  date
            (r"^(0?[1-9]|1[0-2]):[0-5][0-9]$",                              None), // 7.  time 12h
            (r"^[\w,\s-]+\.[A-Za-z]{3}$",                                   None), // 8.  filename
            (r"^[a-z0-9]+(?:-[a-z0-9]+)*$",                                 None), // 9.  slug

            // ── Patterns with extractable literal bigrams ────────────

            // URL with required protocol
            (r"https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", Some(&[
                ("ht", C), ("tt", C), ("tp", C),   // from "http"
                ("ht", S), ("tp", S),               // from "http" skip-1
                (":/", C), ("//", C),               // from "://"
            ])),

            // fn\s+\w+
            (r"fn\s+\w+", Some(&[
                ("fn", C),                          // from "fn"
                ("n ", C),                          // cross-boundary: 'n' → \s starts ' '
            ])),

            // use\s+crate::
            (r"use\s+crate::", Some(&[
                ("us", C), ("se", C), ("ue", S),   // from "use"
                ("cr", C), ("ra", C), ("at", C),   // from "crate"
                ("te", C), ("::", C),
                ("ca", S), ("rt", S), ("ae", S),   // "crate" skip-1
            ])),

            // unwrap\(\)|expect\(
            (r"unwrap\(\)|expect\(", Some(&[
                ("nw", C), ("wr", C), ("ra", C),   // "unwrap("
                ("ap", C), ("p(", C),
                ("xp", C), ("pe", C), ("ec", C),   // "expect("
                ("ct", C), ("t(", C),
            ])),

            // TODO|FIXME|HACK
            (r"TODO|FIXME|HACK", Some(&[
                ("to", C), ("od", C), ("do", C),   // "TODO"
                ("fi", C), ("ix", C), ("xm", C),   // "FIXME"
                ("me", C),
                ("ha", C), ("ac", C), ("ck", C),   // "HACK"
                ("hc", S), ("ak", S),               // "HACK" skip-1
            ])),
        ];

        for (i, &(pattern, expected)) in cases.iter().enumerate() {
            let q = regex_to_bigram_query(pattern);

            if let Some(bigrams) = expected {
                assert!(
                    !q.is_any(),
                    "#{i} {pattern:?}: expected bigrams but got Any"
                );

                for &(pair, skip) in bigrams {
                    let b = pair.as_bytes();
                    debug_assert_eq!(b.len(), 2, "bigram must be 2 chars: {pair:?}");
                    let found = if skip {
                        has_skip1(&q, b[0], b[1])
                    } else {
                        has_consec(&q, b[0], b[1])
                    };
                    let kind = if skip { "skip-1" } else { "consec" };
                    assert!(found, "#{i} {pattern:?}: missing {kind} bigram {pair:?}");
                }
            }
        }
    }

    #[test]
    fn fuzzy_short_query_is_any() {
        assert!(fuzzy_to_bigram_query("", 6).is_any());
        assert!(fuzzy_to_bigram_query("a", 6).is_any());
    }

    #[test]
    fn fuzzy_two_char_query_is_and_of_one() {
        // len 2 → max_typos = 0 → AND of single bigram → simplified to Consec.
        let q = fuzzy_to_bigram_query("ab", 6);
        assert!(matches!(q, BigramQuery::Consec(_)));
    }

    #[test]
    fn fuzzy_six_char_query_filters_real_corpus() {
        // 6-byte needle → max_typos = 2 → OR-of-AND structure.
        // It should still narrow the candidate set on a corpus that
        // contains the literal needle vs. one that doesn't.
        let idx = build_test_index(&[
            b"the quick brown fox", // 0: contains "quick"
            b"a sneaky donkey",     // 1
            b"another quick test",  // 2: contains "quick"
        ]);
        let q = fuzzy_to_bigram_query("quickb", 6);
        assert!(!q.is_any());
        // Should yield a candidate bitset that contains at least item 0.
        // (Items 1 and 2 may or may not also match — the probe-subset
        // logic deliberately admits some false positives.)
        let candidates = q.evaluate(&idx).expect("fuzzy bigram query evaluable");
        assert!(BigramIndex::is_candidate(&candidates, 0));
    }
}
