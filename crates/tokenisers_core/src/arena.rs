use crate::error::{Result, TokeniserError};
use crate::vocab::TokenId;

pub const NULL_NODE: u32 = u32::MAX;

/// Exactly 16-byte SymbolNode, cache-aligned (4 nodes per 64-byte CPU cache line).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolNode {
    pub token: TokenId, // 4 bytes
    pub prev: u32,      // 4 bytes
    pub next: u32,      // 4 bytes
    pub version: u32,   // 4 bytes
}

impl Default for SymbolNode {
    #[inline(always)]
    fn default() -> Self {
        Self {
            token: u32::MAX,
            prev: NULL_NODE,
            next: NULL_NODE,
            version: 0,
        }
    }
}

impl SymbolNode {
    #[inline(always)]
    pub fn new(token: TokenId) -> Self {
        Self {
            token,
            prev: NULL_NODE,
            next: NULL_NODE,
            version: 0,
        }
    }

    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.token != u32::MAX
    }
}

/// Metadata describing an isolated pre-tokenized word chain in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordEntry {
    pub head: u32,
    pub tail: u32,
    pub len: u32,
    pub freq: u64,
}

/// Doubly-linked symbol arena providing O(1) merge splicing and zero memory relocation.
#[derive(Clone, Debug, Default)]
pub struct SymbolArena {
    nodes: Vec<SymbolNode>,
    words: Vec<WordEntry>,
}

impl SymbolArena {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            words: Vec::new(),
        }
    }

    pub fn with_capacity(node_cap: usize, word_cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(node_cap),
            words: Vec::with_capacity(word_cap),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline(always)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline(always)]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    #[inline(always)]
    pub fn nodes(&self) -> &[SymbolNode] {
        &self.nodes
    }

    #[inline(always)]
    pub fn words(&self) -> &[WordEntry] {
        &self.words
    }

    #[inline(always)]
    pub fn get_node(&self, idx: u32) -> Option<&SymbolNode> {
        self.nodes.get(idx as usize)
    }

    #[inline(always)]
    pub fn get_node_mut(&mut self, idx: u32) -> Option<&mut SymbolNode> {
        self.nodes.get_mut(idx as usize)
    }

    #[inline(always)]
    pub fn get_word(&self, word_id: u32) -> Option<&WordEntry> {
        self.words.get(word_id as usize)
    }

    #[inline(always)]
    pub fn get_word_mut(&mut self, word_id: u32) -> Option<&mut WordEntry> {
        self.words.get_mut(word_id as usize)
    }

    #[inline(always)]
    pub fn is_node_active(&self, idx: u32) -> bool {
        if let Some(node) = self.nodes.get(idx as usize) {
            node.is_active()
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn get_token(&self, idx: u32) -> TokenId {
        self.nodes[idx as usize].token
    }

    #[inline(always)]
    pub fn get_version(&self, idx: u32) -> u32 {
        self.nodes[idx as usize].version
    }

    /// Appends a word sequence to the arena, linking adjacent nodes and registering the word.
    pub fn add_word(&mut self, tokens: &[TokenId], freq: u64) -> u32 {
        if tokens.is_empty() {
            let word_id = self.words.len() as u32;
            self.words.push(WordEntry {
                head: NULL_NODE,
                tail: NULL_NODE,
                len: 0,
                freq,
            });
            return word_id;
        }

        let start_idx = self.nodes.len() as u32;
        let count = tokens.len();

        for (i, &tok) in tokens.iter().enumerate() {
            let idx = start_idx + i as u32;
            let prev = if i == 0 { NULL_NODE } else { idx - 1 };
            let next = if i + 1 == count { NULL_NODE } else { idx + 1 };
            self.nodes.push(SymbolNode {
                token: tok,
                prev,
                next,
                version: 0,
            });
        }

        let word_id = self.words.len() as u32;
        self.words.push(WordEntry {
            head: start_idx,
            tail: start_idx + (count as u32 - 1),
            len: count as u32,
            freq,
        });

        word_id
    }

    /// Directly links two nodes in the arena: `left.next = right` and `right.prev = left`.
    /// Used during SuperBPE boundary lifting transitions.
    #[inline]
    pub fn link_nodes(&mut self, left_idx: u32, right_idx: u32) {
        if left_idx != NULL_NODE {
            self.nodes[left_idx as usize].next = right_idx;
        }
        if right_idx != NULL_NODE {
            self.nodes[right_idx as usize].prev = left_idx;
        }
    }

    /// Performs an O(1) pair merge splicing:
    /// Replaces adjacent pair `(left_idx, right_idx)` with `new_token` at `left_idx`.
    /// Tombstones `right_idx`.
    /// Rewires neighbor pointers and increments version counters.
    ///
    /// Returns `(prev_neighbor, next_neighbor)` arena indices around the merged node.
    pub fn splice_pair(
        &mut self,
        left_idx: u32,
        right_idx: u32,
        new_token: TokenId,
        word_id: u32,
    ) -> Result<(u32, u32)> {
        let n_nodes = self.nodes.len();
        if left_idx as usize >= n_nodes {
            return Err(TokeniserError::ArenaIndexOutOfBounds(left_idx, n_nodes));
        }
        if right_idx as usize >= n_nodes {
            return Err(TokeniserError::ArenaIndexOutOfBounds(right_idx, n_nodes));
        }

        let left = self.nodes[left_idx as usize];
        let right = self.nodes[right_idx as usize];

        if left.next != right_idx || right.prev != left_idx {
            return Err(TokeniserError::CorruptedArenaLink(
                left_idx, right_idx, right.prev,
            ));
        }

        let prev_neighbor = left.prev;
        let next_neighbor = right.next;

        // 1. Mutate left node to become the merged token
        self.nodes[left_idx as usize].token = new_token;
        self.nodes[left_idx as usize].next = next_neighbor;
        self.nodes[left_idx as usize].version = left.version.wrapping_add(1);

        // 2. Tombstone right node
        self.nodes[right_idx as usize].prev = NULL_NODE;
        self.nodes[right_idx as usize].next = NULL_NODE;
        self.nodes[right_idx as usize].token = u32::MAX;
        self.nodes[right_idx as usize].version = right.version.wrapping_add(1);

        // 3. Rewire neighbor pointers
        if prev_neighbor != NULL_NODE {
            self.nodes[prev_neighbor as usize].next = left_idx;
        }
        if next_neighbor != NULL_NODE {
            self.nodes[next_neighbor as usize].prev = left_idx;
        }

        // 4. Update word entry metadata
        if let Some(word) = self.words.get_mut(word_id as usize) {
            if word.tail == right_idx {
                word.tail = left_idx;
            }
            if word.head == right_idx {
                word.head = left_idx;
            }
            word.len = word.len.saturating_sub(1);
        }

        Ok((prev_neighbor, next_neighbor))
    }

    /// Extracts active tokens for a specific word.
    pub fn get_word_tokens(&self, word_id: u32) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        if let Some(word) = self.words.get(word_id as usize) {
            let mut curr = word.head;
            while curr != NULL_NODE {
                let node = &self.nodes[curr as usize];
                if node.is_active() {
                    tokens.push(node.token);
                }
                curr = node.next;
            }
        }
        tokens
    }
}
