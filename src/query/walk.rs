//! Verification-pass walks over base + overlay.
//!
//! Both walks are parameterised on a `verify: Fn(&ChunkItem) ->
//! Option<MatchLocation>` closure so the literal, regex, and fuzzy
//! callers can plug in their own verifier without duplicating the
//! loop skeleton.

use crate::index::{BaseIndex, ChunkItem, Overlay};

use super::Hit;
use super::candidate::{CandidateSet, MatchLocation};
use super::snippet::make_hit;

/// Walk the base chunks, applying `verify` to each survivor of the
/// candidate set. Stops once `hits` reaches `limit`.
///
/// The two-arm `Some(bitset)` / `None` pyramid the search functions
/// used to carry collapses to one loop here; the asymmetry between
/// "have prefilter" and "no prefilter" is encapsulated by
/// [`CandidateSet::includes`].
pub(crate) fn walk_base_chunks<F>(
    base: &BaseIndex,
    ov: &Overlay,
    candidates: &CandidateSet,
    limit: usize,
    hits: &mut Vec<Hit>,
    verify: F,
) where
    F: Fn(&ChunkItem) -> Option<MatchLocation>,
{
    for (i, chunk) in base.chunks.iter().enumerate() {
        if !candidates.includes(i, ov) {
            continue;
        }
        if let Some(loc) = verify(chunk) {
            hits.push(make_hit(chunk, loc));
            if hits.len() >= limit {
                break;
            }
        }
    }
}

/// Same as [`walk_base_chunks`] over a list of overflow indices.
pub(crate) fn walk_overflow<F>(
    ov: &Overlay,
    indices: Vec<usize>,
    limit: usize,
    hits: &mut Vec<Hit>,
    verify: F,
) where
    F: Fn(&ChunkItem) -> Option<MatchLocation>,
{
    let chunks = ov.overflow.chunks();
    for idx in indices {
        let chunk = &chunks[idx];
        if let Some(loc) = verify(chunk) {
            hits.push(make_hit(chunk, loc));
            if hits.len() >= limit {
                break;
            }
        }
    }
}
