//! [`CandidateSet`] and [`MatchLocation`] — the two small types every
//! search mode shares.
//!
//! `CandidateSet` encapsulates "do we have a bigram prefilter, and if
//! so, what survived after AND-NOTing the tombstones?" so the
//! verification walk is one loop instead of two near-identical branches.
//! `MatchLocation` records *which* string (`text_norm_ascii` vs
//! `text_utf8`) a verifier matched in, so the snippet renderer can pick
//! the right offset-mapping strategy.

use crate::bitset::Bitset;
use crate::index::Overlay;

/// Where a match was found inside a chunk.
///
/// The two variants exist because literal / fuzzy search locate matches
/// in `text_norm_ascii` (the normalised byte string used for the bigram
/// prefilter and `memchr`), while regex search locates them directly in
/// `text_utf8` (the original UTF-8 the regex engine ran against). The
/// snippet renderer needs to know which: an in-norm offset must be
/// proportionally remapped to a byte offset in `text_utf8` because the
/// two strings diverge after deunicode + lowercase + whitespace
/// collapse; an in-utf8 offset is already exact.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MatchLocation {
    /// Byte offset within `text_norm_ascii`.
    Norm { offset: usize, query_len: usize },
    /// Byte offset within `text_utf8`.
    Utf8 { offset: usize, match_len: usize },
}

/// Candidate set fed into the verification pass.
///
/// Either:
/// * `Unconstrained` — the bigram prefilter had nothing to say (no
///   bigrams in the query, or `BigramIndex::query` returned `None`).
///   The verifier must visit every base chunk that isn't tombstoned.
/// * `Restricted(Bitset)` — a bigram bitset already AND-NOTed with
///   the overlay's tombstones at construction.
///
/// Hiding the two cases behind one type lets the base-walk be a single
/// loop over `[0, base.chunks.len())` instead of two near-identical
/// match arms.
pub(crate) enum CandidateSet {
    Unconstrained,
    Restricted(Bitset),
}

impl CandidateSet {
    /// Lift the bigram prefilter's `Option<Vec<u64>>` into a
    /// [`CandidateSet`] sized to `base_chunk_count`.
    ///
    /// `None` ⇒ `Unconstrained`; `Some(words)` ⇒ `Restricted` over
    /// exactly `base_chunk_count` bits, with the overlay's tombstones
    /// masked out in the same step.
    pub(crate) fn from_bigram_lookup(
        lookup: Option<Vec<u64>>,
        base_chunk_count: usize,
        ov: &Overlay,
    ) -> Self {
        match lookup {
            None => Self::Unconstrained,
            Some(words) => {
                let mut bits = Bitset::from_words(words, base_chunk_count);
                bits.and_not_assign(&ov.tombstones);
                Self::Restricted(bits)
            }
        }
    }

    /// Should chunk index `i` be visited by the verifier?
    ///
    /// `Restricted`: check the bitset. Tombstones are already masked
    /// out at construction, so a single lookup suffices.
    /// `Unconstrained`: skip tombstoned indices; everything else passes.
    #[inline]
    pub(crate) fn includes(&self, i: usize, ov: &Overlay) -> bool {
        match self {
            Self::Restricted(bits) => bits.get(i),
            Self::Unconstrained => !ov.is_tombstoned(i),
        }
    }
}
