#############################################################
#            Data Science Capstone: project Movielens
#                   Alireza Jafari Arimi
#############################################################

# Note: this process could take a couple of minutes for loading required package: 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library('ggplot2')
library('dplyr')
library('knitr')
library('stringr')
library('DT')
library('kableExtra')

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
# data loading
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
ratings <- as.data.frame(str_split(readLines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(readLines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)
# Exploratory Data Analysis (EDA)
# data structure 
str(edx,7)
#overview of data set
 datatable(head(edx, 7))
#Summary of data Statistics
summary_edx <- summary(edx)
kable(summary_edx, caption = "Summary Statistics") %>%
  kable_styling(position = "center", latex_options = c("scale_down", "striped"))
#Search for missing values 
colSums(is.na(edx))
anyNA(edx)
# Visualization and data analysis 
# Number of unique values
distinct <- edx %>% summarize(n_users = n_distinct(userId),           
                              n_movies = n_distinct(movieId), n_genres = n_distinct(genres)) 
kable(distinct, caption = "number of unique users, movies, and genres") %>%
  kable_styling(position = "center", latex_options = c("scale_down", "striped"))
# unique rating
unique(edx$rating)
#distribution Rating Values
#Top 20 movies titles based on rating. 
#bar chart of top 20 movies based on rating 
#  calculate the number of ratings for each movie
movie_ratings <- edx %>%
group_by(title) %>%
summarise(count = n(), .groups = "drop")
#  top 20 movies title based on the number of ratings
top_movies <- movie_ratings %>%   top_n(20, count)
# Create the bar chart (invers cone) of top 20 movies
ggplot(top_movies, aes(x=reorder(title, -count), y=count)) +   geom_bar(stat='identity', fill="blue") +   coord_flip() +     labs(x="", y="Number of ratings") +  geom_text(aes(label=count), hjust=-0.1, size=3) +   labs(title="Top 20 movies title based on number of ratings", caption = "source data: edx set")
#  top 20 movies based on genres
top_genres <- edx %>%
  group_by(genres) %>%
  summarize(n = n(), .groups = "drop") %>%
  arrange(desc(n))
# Select the top 20 genres
top_20_genres <- top_genres %>%   top_n(20)
# Create the plot
ggplot(top_20_genres, aes(x = reorder(genres, n), y = n)) +
  geom_bar(stat = "identity") +   coord_flip() +   labs(x = "Genres", y = "Movie Count", title = "Top 20 Genres by Movie Count") +
  theme_minimal()
# Calculate the number of movies percentage per genre
top_genres <- edx %>%
  group_by(genres) %>%
  summarize(n = n(), .groups = "drop") %>%
  mutate(total = sum(n), percentage = n / total * 100) %>%
  arrange(desc(percentage))
# Select the top 20 genres
top_20_genres <- top_genres %>%
  top_n(20, percentage)
# Create the plot
ggplot(top_20_genres, aes(x = reorder(genres, percentage), y = percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Genres", y = "Percentage", title = "Top 20 Genres by Movie Percentage") +
  theme_minimal()
# top 20 movie genres by average rating 
#Distribution of ratings by movieID
# Number of rating by movieID
edx %>%
  group_by(movieId) %>%
  summarize(n = n(), .groups = "drop") %>%
  ggplot(aes(n)) +   geom_histogram(bins = 30, color = "blue") +   scale_x_log10() +
  xlab("Number of ratings") +   ylab("Number of movies") +   ggtitle("Number of ratings given by movieID")
#Distribution of rating by user ID
edx %>% 
  group_by(userId) %>%   summarize(n = n(), .groups = "drop") %>%
  ggplot(aes(n)) +   geom_histogram(bins = 30, color = "blue") +   scale_x_log10() +
  xlab("Number of ratings") +   ylab("Number of users") +   ggtitle("Number of ratings given by users")
# edx dtat spliting for training and test set

# Final hold-out test set will be 10% of edx data set as mentioned in the course and will be served as  validation data set.
validation_set <- final_holdout_test
# creating training/test(80/20)  set
#creating an index to pick a sample for training and test set
set.seed(123)
edx_test_index <- createDataPartition(y = edx$rating,
                                      times = 1, p = 0.2, list = FALSE)
training_set <- edx[-edx_test_index, ] #creating the training set
test_set <- edx[edx_test_index, ] #creating the test set
# Creating RMSE Function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2,
            na.rm = TRUE))
}
# First and simple modelbased on overal mean rating
mu <- mean(training_set$rating)
mu
# RMSE1 value 
RMSE1 <- RMSE (test_set$rating, mu)
RMSE1 

# Building second model using Movie bias(effects)
bm <- training_set %>%
group_by(movieId) %>%
summarize(bm = mean(rating - mu))
ggplot(bm, aes(x = bm)) +
geom_histogram(bins = 20, fill = "green", color = "black") +
labs(title = "Distribution of movie bias", x = "Movie Bias(bm)", y = "Number of movies") +
theme_minimal()
# Join bm to the test set
test_set <- test_set %>%
  left_join(bm, by = "movieId")
# Predict ratings as mu + bm
bm_prediction <- mu + test_set$bm
# Calculate RMSE on the ratings
RMSE_movies <- RMSE(test_set$rating, bm_prediction)
RMSE_movies
# creating RSME table
RMSE_table <- data.frame( RMSE = c(RMSE1, RMSE_movies),   Stage = c("Rating model ", " Movie effects model ") )
RMSE_table %>%
  kable("html") %>%
  kable_styling("responsive")
# Calculate user(bias) average rating as bu
bu <- training_set %>% left_join(bm, by = "movieId") %>% group_by(userId) %>%
  summarize(bu = mean(rating - mu - bm))
# Create plot(histogram) of user bias
ggplot(bu, aes(x = bu)) + 
  geom_histogram(bins = 20, fill = "green", color = "black") +      labs(title = " Distribution of User Bias ", x = "bu", y = "Number of movies") +
  theme_minimal()
# Construct predictors and improves RMSE
bu_prediction <- validation_set %>%
  left_join(bm, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + bm + bu) %>%
  .$pred
# Calculation of RMSEs
RSME_user_movies <- RMSE(bu_prediction, validation_set$rating)
RSME_user_movies

# Creating RMSEs table
RMSE_table <- data.frame(   RMSE = c(RMSE1, RMSE_movies,RSME_user_movies)
                            , Stage = c("Average rating model", "Movie effects model","Movie_User effects model" )  )
RMSE_table %>%
  kable("html") %>% kable_styling("responsive")

# regualization and tuning with lambda
# Create a sequence of values for lambda ranging from 0 to 10 with 0.25 increments
lambda <- seq(0, 10, 0.25)
# Apply and validate the regularized model against the validation data set
RMSES <- sapply(lambda, function(l){
    bm <- edx %>%
    group_by(movieId) %>%
    summarise(bm = sum(rating - mu)/(n()+l))
    bu <- edx %>%
    left_join(bm, by="movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bm - mu)/(n()+l))
  # Ratings prediction by validation set
    predicted_ratings <- validation_set %>%
    left_join(bm, by="movieId") %>%
    left_join(bu, by="userId") %>%
    mutate(pred = mu + bm + bu) %>%
    pull(pred)
  # Calculate RMSE and evaluate accuracy
  return( RMSE(validation_set$rating, predicted_ratings))
})
RMSE_REG_MOVIE_USER <- min(RMSES)
RMSE_REG_MOVIE_USER 
# Plot rmses vs lambdas to select the optimal omega                                                             
qplot(lambda, RMSES,  color = I("#51A8FF"), 
      main = "RMSEs vs. Lambdas") + 
  theme_minimal()

# Lambda that minimizes RMSEs for MOVIE + USER
lambda <- lambda[which.min(RMSES)] 
lambda

# RMSE table of results
RMSE_table <- data.frame(   RMSE = c(RMSE1, RMSE_movies,RSME_user_movies, RMSE_REG_MOVIE_USER)
                            , Stage = c("Average rating model", "Movie effects model","Movie_User effects model","Rg.Movie_User effects model" )  )
RMSE_table %>%
  kable("html") %>% kable_styling("responsive")
