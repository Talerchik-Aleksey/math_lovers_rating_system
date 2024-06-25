extern crate nalgebra as na;

use na::{DMatrix, SVD, Dynamic};
use std::collections::HashMap;

type UserId = u32;
type MovieId = u32;
type Rating = f32;
type Ratings = HashMap<UserId, HashMap<MovieId, Rating>>;

fn ratings_to_matrix(ratings: &Ratings, num_users: usize, num_movies: usize) -> DMatrix<f32> {
    let mut matrix = DMatrix::zeros(num_users, num_movies);

    for (user, user_ratings) in ratings {
        for (movie, &rating) in user_ratings {
            matrix[(user.clone() as usize - 1, movie.clone() as usize - 1)] = rating;
        }
    }

    matrix
}

fn predict_rating(svd: &SVD<f32, Dynamic, Dynamic>, user: usize, movie: usize) -> f32 {
    let u = svd.u.as_ref().unwrap();
    let sigma = DMatrix::from_diagonal(&svd.singular_values);
    let v_t = svd.v_t.as_ref().unwrap();

    let reconstructed = u * sigma * v_t;
    reconstructed[(user, movie)]
}

fn ratings_to_matrix_with_bias(ratings: &Ratings, num_users: usize, num_movies: usize) -> (DMatrix<f32>, Vec<f32>) {
    let mut matrix = DMatrix::zeros(num_users, num_movies);
    let mut user_means = Vec::with_capacity(num_users);

    for (user, user_ratings) in ratings {
        let mean_rating: f32 = user_ratings.values().sum::<f32>() / user_ratings.len() as f32;
        user_means.push(mean_rating);

        for (movie, &rating) in user_ratings {
            matrix[(user.clone() as usize - 1, movie.clone() as usize - 1)] = rating - mean_rating;
        }
    }

    (matrix, user_means)
}

fn predict_rating_with_bias(svd: &SVD<f32, Dynamic, Dynamic>, user_means: &Vec<f32>, user: usize, movie: usize) -> f32 {
    let u = svd.u.as_ref().unwrap();
    let sigma = DMatrix::from_diagonal(&svd.singular_values);
    let v_t = svd.v_t.as_ref().unwrap();

    let reconstructed = u * sigma * v_t;
    reconstructed[(user, movie)] + user_means[user]
}

fn main() {
    let mut ratings: Ratings = HashMap::new();
    ratings.insert(1, vec![(1, 25.0), (2, 13.0), (3, 50.0)].into_iter().collect());
    ratings.insert(2, vec![(1, 34.0), (2, 15.0), (3, 30.0)].into_iter().collect());
    ratings.insert(3, vec![(1, 55.0), (2, 90.0)].into_iter().collect());
    ratings.insert(4, vec![(2, 100.0)].into_iter().collect());

    let num_users = ratings.len();
    let num_movies = ratings.values().map(|user_ratings| user_ratings.len()).max().unwrap_or(0);

     let (matrix, user_means) = ratings_to_matrix_with_bias(&ratings, num_users, num_movies);
    let svd = SVD::new(matrix, true, true);

    let predicted_rating = predict_rating_with_bias(&svd, &user_means, 2, 2);
    println!("Predicted rating for user 3 for movie 3: {}", predicted_rating);
}

