use ahash::AHashMap;
use crate::arena::{SymbolArena, NULL_NODE};
use crate::vocab::TokenId;

/// Adjacent symbol pair representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair(pub TokenId, pub TokenId);

impl Pair {
    #[inline(always)]
    pub fn new(left: TokenId, right: TokenId) -> Self {
        Self(left, right)
    }

    #[inline(always)]
    pub fn to_u64(self) -> u64 {
        ((self.0 as u64) << 32) | (self.1 as u64)
    }

    #[inline(always)]
    pub fn from_u64(val: u64) -> Self {
        Self((val >> 32) as TokenId, val as TokenId)
    }
}

/// A reference to an occurrence of a symbol pair in the arena, with versioned tombstoning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccurrenceRef {
    pub word_id: u32,
    pub node_idx: u32,
    pub version: u32,
}

impl OccurrenceRef {
    #[inline(always)]
    pub fn new(word_id: u32, node_idx: u32, version: u32) -> Self {
        Self {
            word_id,
            node_idx,
            version,
        }
    }

    /// Verifies in O(1) whether this occurrence is still structurally valid in the arena.
    #[inline(always)]
    pub fn is_valid(&self, expected_pair: Pair, arena: &SymbolArena) -> bool {
        if let Some(left) = arena.get_node(self.node_idx) {
            if left.version == self.version && left.token == expected_pair.0 && left.next != NULL_NODE {
                if let Some(right) = arena.get_node(left.next) {
                    return right.token == expected_pair.1;
                }
            }
        }
        false
    }
}

/// Inverted index tracking occurrences of symbol pairs across the arena with lazy version tombstoning.
#[derive(Clone, Debug, Default)]
pub struct InvertedIndex {
    occurrences: AHashMap<Pair, Vec<OccurrenceRef>>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            occurrences: AHashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            occurrences: AHashMap::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn add_occurrence(&mut self, pair: Pair, occ: OccurrenceRef) {
        self.occurrences.entry(pair).or_default().push(occ);
    }

    #[inline(always)]
    pub fn get(&self, pair: &Pair) -> Option<&Vec<OccurrenceRef>> {
        self.occurrences.get(pair)
    }

    #[inline(always)]
    pub fn take(&mut self, pair: &Pair) -> Option<Vec<OccurrenceRef>> {
        self.occurrences.remove(pair)
    }

    #[inline(always)]
    pub fn contains(&self, pair: &Pair) -> bool {
        self.occurrences.contains_key(pair)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.occurrences.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.occurrences.is_empty()
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.occurrences.clear();
    }

    #[inline(always)]
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, Pair, Vec<OccurrenceRef>> {
        self.occurrences.iter()
    }
}
