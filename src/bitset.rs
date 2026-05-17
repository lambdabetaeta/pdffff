//! Dense bitset indexed by item position.
//!
//! A small, focused type used wherever the codebase needs "is bit `i`
//! set in a bitmap covering `n` items": tombstones over base-index
//! chunks, and bigram-prefilter candidate sets returned from
//! [`crate::bigram::BigramIndex::query`].
//!
//! The bitset stores its logical length in bits so consumers can iterate
//! by item index without padding-bit confusion; the trailing bits of the
//! final word (if `n % BITS_PER_WORD != 0`) are always zero and any
//! operation that mutates them is required to preserve that invariant.

/// Width of the bitset's word in bits.
pub const BITS_PER_WORD: usize = u64::BITS as usize;

/// `(word_index, bit_mask)` for bit `i`.
#[inline]
fn position(i: usize) -> (usize, u64) {
    (i / BITS_PER_WORD, 1u64 << (i % BITS_PER_WORD))
}

/// Words needed to cover `bits` bits.
#[inline]
pub fn words_for(bits: usize) -> usize {
    bits.div_ceil(BITS_PER_WORD)
}

/// Fixed-length dense bitset.
#[derive(Debug, Clone)]
pub struct Bitset {
    words: Vec<u64>,
    /// Number of valid bits (≤ `words.len() * BITS_PER_WORD`).
    len: usize,
}

impl Bitset {
    /// All-zero bitset covering `len` bits.
    pub fn zeros(len: usize) -> Self {
        Self {
            words: vec![0u64; words_for(len)],
            len,
        }
    }

    /// Wrap `words` as a bitset of `len` bits.
    ///
    /// Used to adopt bitsets handed back by other crates (the bigram
    /// prefilter returns `Vec<u64>` for FFI / call-overhead reasons).
    /// Panics in debug builds if `words` is sized inconsistently with `len`.
    pub fn from_words(words: Vec<u64>, len: usize) -> Self {
        debug_assert_eq!(
            words.len(),
            words_for(len),
            "bitset word count must match ceil(len / BITS_PER_WORD)",
        );
        Self { words, len }
    }

    /// Logical bit length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw word view for hot paths that need slab access.
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Set bit `i`. Silently no-op if `i >= len` (a caller bug, but
    /// the alternative — panicking — would crash the watcher thread).
    #[inline]
    pub fn set(&mut self, i: usize) {
        let (w, m) = position(i);
        if w < self.words.len() {
            self.words[w] |= m;
        }
    }

    /// True iff bit `i` is set.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        let (w, m) = position(i);
        w < self.words.len() && self.words[w] & m != 0
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// `self &= !other`, word-wise, over `min(self.words, other.words)`.
    ///
    /// Mismatched-length bitsets are allowed: words past either side's
    /// end are left untouched. This is the AND-NOT used by the query
    /// engine to mask tombstoned candidates out of a bigram bitset.
    pub fn and_not_assign(&mut self, other: &Bitset) {
        let n = self.words.len().min(other.words.len());
        for w in 0..n {
            self.words[w] &= !other.words[w];
        }
    }
}

impl Default for Bitset {
    fn default() -> Self {
        Self::zeros(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_then_set_then_get() {
        let mut b = Bitset::zeros(130);
        assert_eq!(b.len(), 130);
        assert!(!b.get(0));
        b.set(0);
        b.set(63);
        b.set(64);
        b.set(129);
        assert!(b.get(0));
        assert!(b.get(63));
        assert!(b.get(64));
        assert!(b.get(129));
        assert!(!b.get(128));
        assert_eq!(b.count_ones(), 4);
    }

    #[test]
    fn set_out_of_range_is_noop() {
        let mut b = Bitset::zeros(10);
        b.set(1000);
        assert_eq!(b.count_ones(), 0);
        assert!(!b.get(1000));
    }

    #[test]
    fn and_not_masks_in_place() {
        let mut a = Bitset::from_words(vec![0b1111u64], 4);
        let b = Bitset::from_words(vec![0b0110u64], 4);
        a.and_not_assign(&b);
        assert!(a.get(0));
        assert!(!a.get(1));
        assert!(!a.get(2));
        assert!(a.get(3));
    }

    #[test]
    fn and_not_tolerates_short_other() {
        let mut a = Bitset::zeros(128);
        a.set(0);
        a.set(64);
        let b = Bitset::from_words(vec![0b1u64], 1); // 1 word, covers bit 0
        a.and_not_assign(&b);
        assert!(!a.get(0));
        // bit 64 is in the second word; b is too short to touch it.
        assert!(a.get(64));
    }

    #[test]
    fn words_for_rounds_up() {
        assert_eq!(words_for(0), 0);
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(64), 1);
        assert_eq!(words_for(65), 2);
        assert_eq!(words_for(128), 2);
    }
}
