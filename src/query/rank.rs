//! Cheap deterministic post-ranking applied to hit lists.

use super::Hit;

/// Cheap deterministic ordering for hits.
///
/// Sort by, in order:
/// 1. Exact phrase hit before partial-term hit (treat the normalized
///    query as a phrase).
/// 2. More matched terms (whitespace-split) before fewer.
/// 3. Earlier match offset before later.
/// 4. Lower `page_no` before higher.
/// 5. Stable: `doc_id` then `chunk_id`.
pub fn cheap_rank(hits: &mut Vec<Hit>, query_norm: &str) {
    let phrase = query_norm.trim();
    let terms: Vec<&str> = phrase.split_whitespace().collect();
    // Pre-compute the lowercased snippet once per hit to avoid
    // re-lowercasing inside the comparator.
    let snippets_lc: Vec<String> = hits.iter().map(|h| h.snippet.to_lowercase()).collect();

    let keys: Vec<(bool, usize, usize, u32)> = hits
        .iter()
        .zip(snippets_lc.iter())
        .map(|(h, snip_lc)| {
            let has_phrase = !phrase.is_empty() && snip_lc.contains(phrase);
            let term_count = terms.iter().filter(|t| snip_lc.contains(**t)).count();
            // Earlier offset of the phrase in the snippet (saturating to
            // a large sentinel when not present so phrase-bearing hits
            // win the tiebreak).
            let offset = if has_phrase {
                snip_lc.find(phrase).unwrap_or(usize::MAX)
            } else if let Some(t) = terms.iter().filter_map(|t| snip_lc.find(t)).min() {
                t
            } else {
                usize::MAX
            };
            (has_phrase, term_count, offset, h.page_no)
        })
        .collect();

    let mut indices: Vec<usize> = (0..hits.len()).collect();
    indices.sort_by(|&a, &b| {
        let (pa, ta, oa, ga) = keys[a];
        let (pb, tb, ob, gb) = keys[b];
        pb.cmp(&pa)
            .then_with(|| tb.cmp(&ta))
            .then_with(|| oa.cmp(&ob))
            .then_with(|| ga.cmp(&gb))
            .then_with(|| hits[a].doc_id.cmp(&hits[b].doc_id))
            .then_with(|| hits[a].chunk_id.cmp(&hits[b].chunk_id))
    });

    let reordered: Vec<Hit> = indices.into_iter().map(|i| hits[i].clone()).collect();
    *hits = reordered;
}
