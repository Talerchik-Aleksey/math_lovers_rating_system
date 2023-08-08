mod libs;
use std::collections::HashMap;

use libs::cosine_similaratiy;

// Define type aliases for clarity
type UserId = u32;
type MovieId = u32;
type Rating = f32;

// User ID -> (Movie ID -> Rating)
type Ratings = HashMap<UserId, HashMap<MovieId, Rating>>;

// Predict a user's rating for a movie
fn predict_rating(ratings: &Ratings, user: UserId, movie: MovieId) -> f32 {
    let mut total_similarity = 0.0;
    let mut weighted_ratings = 0.0;

    // Compute the mean rating for the target user
    let user_ratings = ratings.get(&user).unwrap();
    let mean_user_rating: Rating = user_ratings.values().sum::<Rating>() / user_ratings.len() as f32;

    for (other_user, other_ratings) in ratings {
        if other_user != &user && other_ratings.contains_key(&movie) {
            let similarity = cosine_similaratiy::cosine_similarity(user_ratings, other_ratings);

            // Compute the mean rating for the other user
            let mean_other_rating: Rating = other_ratings.values().sum::<Rating>() / other_ratings.len() as f32;

            // Normalize the other user's rating by subtracting their mean rating
            let normalized_other_rating = other_ratings.get(&movie).unwrap() - mean_other_rating;

            total_similarity += similarity.abs();
            weighted_ratings += similarity * normalized_other_rating;
        }
    }

    // Add the target user's mean rating back to the prediction
    mean_user_rating + weighted_ratings / total_similarity
}


fn main() {
    // Define some dummy data
    let mut ratings: Ratings = HashMap::new();

    // User 1 ratings
    let mut user1_ratings: HashMap<MovieId, Rating> = HashMap::new();
    user1_ratings.insert(1, 1.0);
    user1_ratings.insert(2, 1.0);
    user1_ratings.insert(3, 5.0);
    ratings.insert(1, user1_ratings);

    // User 2 ratings
    let mut user2_ratings: HashMap<MovieId, Rating> = HashMap::new();
    user2_ratings.insert(1, 3.0);
    user2_ratings.insert(2, 1.0);
    user2_ratings.insert(3, 4.0);
    ratings.insert(2, user2_ratings);


    // User 3 ratings
    let mut user3_ratings: HashMap<MovieId, Rating> = HashMap::new();
    user3_ratings.insert(1, 3.0);
    user3_ratings.insert(2, 2.0);
    ratings.insert(3, user3_ratings);

    // User 4 ratings
    let mut user2_ratings: HashMap<MovieId, Rating> = HashMap::new();
    user2_ratings.insert(2, 5.0);
    ratings.insert(4, user2_ratings);

    // Predict rating for user 3 for movie 3
    let predicted_rating = predict_rating(&ratings, 3, 3);
    println!("Predicted rating for user 3 for movie 3: {}", predicted_rating);
}

