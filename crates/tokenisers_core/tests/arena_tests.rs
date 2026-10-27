use tokenisers_core::arena::{SymbolArena, SymbolNode, NULL_NODE};
use tokenisers_core::index::{OccurrenceRef, Pair};
use tokenisers_core::vocab::TokenId;

#[test]
fn test_symbol_node_size_is_16_bytes() {
    assert_eq!(std::mem::size_of::<SymbolNode>(), 16);
    assert_eq!(std::mem::align_of::<SymbolNode>(), 4);
}

#[test]
fn test_word_insertion_and_initial_links() {
    let mut arena = SymbolArena::new();
    let tokens: Vec<TokenId> = vec![10, 20, 30, 40];
    let word_id = arena.add_word(&tokens, 5);

    let word = arena.get_word(word_id).expect("Word must exist");
    assert_eq!(word.len, 4);
    assert_eq!(word.freq, 5);
    assert_eq!(word.head, 0);
    assert_eq!(word.tail, 3);

    // Check node pointers
    assert_eq!(arena.get_node(0).unwrap().prev, NULL_NODE);
    assert_eq!(arena.get_node(0).unwrap().next, 1);
    assert_eq!(arena.get_node(0).unwrap().token, 10);

    assert_eq!(arena.get_node(1).unwrap().prev, 0);
    assert_eq!(arena.get_node(1).unwrap().next, 2);
    assert_eq!(arena.get_node(1).unwrap().token, 20);

    assert_eq!(arena.get_node(3).unwrap().prev, 2);
    assert_eq!(arena.get_node(3).unwrap().next, NULL_NODE);
    assert_eq!(arena.get_node(3).unwrap().token, 40);
}

#[test]
fn test_splice_pair_in_middle() {
    // [10 <-> 20 <-> 30 <-> 40]
    let mut arena = SymbolArena::new();
    let word_id = arena.add_word(&[10, 20, 30, 40], 1);

    // Merge (20, 30) -> 250 at indices (1, 2)
    let (prev_idx, next_idx) = arena.splice_pair(1, 2, 250, word_id).expect("Splice failed");
    assert_eq!(prev_idx, 0);
    assert_eq!(next_idx, 3);

    // Node 1 should now be token 250, next pointing to 3, version incremented
    let node1 = arena.get_node(1).unwrap();
    assert_eq!(node1.token, 250);
    assert_eq!(node1.prev, 0);
    assert_eq!(node1.next, 3);
    assert_eq!(node1.version, 1);

    // Node 2 should be tombstoned
    let node2 = arena.get_node(2).unwrap();
    assert!(!node2.is_active());
    assert_eq!(node2.prev, NULL_NODE);
    assert_eq!(node2.next, NULL_NODE);
    assert_eq!(node2.version, 1);

    // Node 0 next should point to 1
    assert_eq!(arena.get_node(0).unwrap().next, 1);
    // Node 3 prev should point to 1
    assert_eq!(arena.get_node(3).unwrap().prev, 1);

    // Active tokens in word should now be [10, 250, 40]
    assert_eq!(arena.get_word_tokens(word_id), vec![10, 250, 40]);
    assert_eq!(arena.get_word(word_id).unwrap().len, 3);
}

#[test]
fn test_splice_pair_at_head() {
    // [10 <-> 20 <-> 30]
    let mut arena = SymbolArena::new();
    let word_id = arena.add_word(&[10, 20, 30], 1);

    // Merge (10, 20) -> 120 at indices (0, 1)
    let (prev_idx, next_idx) = arena.splice_pair(0, 1, 120, word_id).expect("Splice failed");
    assert_eq!(prev_idx, NULL_NODE);
    assert_eq!(next_idx, 2);

    let word = arena.get_word(word_id).unwrap();
    assert_eq!(word.head, 0);
    assert_eq!(word.len, 2);
    assert_eq!(arena.get_word_tokens(word_id), vec![120, 30]);
}

#[test]
fn test_splice_pair_at_tail() {
    // [10 <-> 20 <-> 30]
    let mut arena = SymbolArena::new();
    let word_id = arena.add_word(&[10, 20, 30], 1);

    // Merge (20, 30) -> 230 at indices (1, 2)
    let (prev_idx, next_idx) = arena.splice_pair(1, 2, 230, word_id).expect("Splice failed");
    assert_eq!(prev_idx, 0);
    assert_eq!(next_idx, NULL_NODE);

    let word = arena.get_word(word_id).unwrap();
    assert_eq!(word.tail, 1);
    assert_eq!(word.len, 2);
    assert_eq!(arena.get_word_tokens(word_id), vec![10, 230]);
}

#[test]
fn test_versioned_tombstone_validation() {
    let mut arena = SymbolArena::new();
    let word_id = arena.add_word(&[b'a' as TokenId, b'a' as TokenId, b'a' as TokenId, b'a' as TokenId], 1);
    // Nodes 0, 1, 2, 3 all 'a'

    let occ0 = OccurrenceRef::new(word_id, 0, 0);
    let occ1 = OccurrenceRef::new(word_id, 1, 0);
    let occ2 = OccurrenceRef::new(word_id, 2, 0);
    let pair = Pair::new(b'a' as TokenId, b'a' as TokenId);

    assert!(occ0.is_valid(pair, &arena));
    assert!(occ1.is_valid(pair, &arena));
    assert!(occ2.is_valid(pair, &arena));

    // Merge nodes 0 and 1 into 'aa' (token 300)
    arena.splice_pair(0, 1, 300, word_id).expect("Splice 0-1 failed");

    // occ0 is no longer valid for ('a', 'a') because node 0 is now 300
    assert!(!occ0.is_valid(pair, &arena));

    // occ1 was registered at node 1 with version 0. Node 1 is tombstoned (ver 1).
    // is_valid MUST return false in O(1)!
    assert!(!occ1.is_valid(pair, &arena));

    // occ2 was at node 2. Node 2 is still 'a' with version 0, and node 3 is 'a'.
    assert!(occ2.is_valid(pair, &arena));
}
