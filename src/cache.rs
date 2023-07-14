// Copyright (C) myl7
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::RwLock;

use crate::Prg;

pub trait Cache<K, V> {
    fn get(&self, key: &K) -> Option<&V>;
    fn set(&mut self, key: K, value: V);
}

impl<K, V> Cache<K, V> for () {
    fn get(&self, _: &K) -> Option<&V> {
        None
    }

    fn set(&mut self, _: K, _: V) {}
}

impl<K, V> Cache<K, V> for HashMap<K, V>
where
    K: Hash + Eq,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }

    fn set(&mut self, key: K, value: V) {
        HashMap::insert(self, key, value);
    }
}

pub struct CachedPrg<const LAMBDA: usize, PrgT, C>
where
    PrgT: Prg<LAMBDA>,
    C: Cache<[u8; LAMBDA], [([u8; LAMBDA], [u8; LAMBDA], bool); 2]>,
{
    prg: PrgT,
    cache: RwLock<C>,
}

impl<const LAMBDA: usize, PrgT, C> CachedPrg<LAMBDA, PrgT, C>
where
    PrgT: Prg<LAMBDA>,
    C: Cache<[u8; LAMBDA], [([u8; LAMBDA], [u8; LAMBDA], bool); 2]>,
{
    pub fn new(prg: PrgT, cache: C) -> Self {
        Self {
            prg,
            cache: RwLock::new(cache),
        }
    }
}

impl<const LAMBDA: usize, PrgT, C> Prg<LAMBDA> for CachedPrg<LAMBDA, PrgT, C>
where
    PrgT: Prg<LAMBDA>,
    C: Cache<[u8; LAMBDA], [([u8; LAMBDA], [u8; LAMBDA], bool); 2]>,
{
    fn gen(&self, seed: &[u8; LAMBDA]) -> [([u8; LAMBDA], [u8; LAMBDA], bool); 2] {
        if let Some(result) = self.cache.read().unwrap().get(seed) {
            return result.to_owned();
        }
        let result = self.prg.gen(seed);
        self.cache.write().unwrap().set(seed.to_owned(), result);
        result
    }
}
