use std::collections::HashMap;

use crate::{MovieId, Rating};

pub fn cosine_similarity(a: &HashMap<MovieId, Rating>, b: &HashMap<MovieId, Rating>) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (movie, rating_a) in a {
        if let Some(rating_b) = b.get(movie) {
            dot_product += rating_a * rating_b;
            norm_a += rating_a.powi(2);
            norm_b += rating_b.powi(2);
        }
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

