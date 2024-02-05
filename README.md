# Using Machine Learning in Python to predict if a movie will be a success
A movie is defined as "successful" if its revenue > 2.5 times its budget.  This code attempts to determine if Machine Learning can be used to predict if a movie will be successful based on several input parameters (Ex. the genre, what month it was released, its budget, its duration, actors/directors/writers, etc.)

I used a dataset from the Internet Movie Database of 86,000 movies across multiple countries, from 1894 - 2019.  The final model predicts with **81% accuracy** whether or not a movie will be a success.

![Confusion Matrix](https://user-images.githubusercontent.com/64739529/120267879-af11f680-c259-11eb-96f4-53a9ab09ed0e.png)

All code and the step-by-step processes (with graphs) can be found in my [Jupyter Notebook](https://github.com/krochkind/IMDB-ML/blob/main/IMDB.ipynb)

## Munging the Data Set
1. Remove all rows without budget and worlwide_gross_income [sic]
2. Remove all rows where the budget isn't in $ (USD)
3. Remove $ and , from numeric columns
4. Remove movies with budget < $10,000
5. Extract month from the date_published field<br />(We can't use the year or day a movie was released to help predict future movies, but the month is useful)
6. Drop columns I don't need 
7. Drop remaining rows with NAN data (since there were only a few left)
8. Set dependent variable (Success) to be any movie that grossed > 2.5x its budget
9. Drop variables subject to hindsight bias<br />ex. worlwide_gross_income [sic], votes, reviews, etc.
10. Convert month into 12 dummy variables
11. Convert genre (comma-separated strings) into dummy variables
12. Chunk Actor/Writer/Director into Low/Medium/High Success based on how many successful movies they had been in<br />This was because there are 64,000 actors, so creating separate dummy variables for each was impractical<br /><br />

## Algorithms
I compared 15 transformation algorithms and 40 classification algorithms, based on 5-Fold Cross Validation testing accuracy.  The best results were found with **QuantileTransformer** and **BaggingClassifier-SVC**.  I then used GridSearchCV to tune the hyperparameters.

## Limitations to the model
Unfortunately, the data I used was subject to "hindsight bias" because the model was trained using the current measure of success.  For example, I classified Harrison Ford as a "High Success" actor, so according to the data, *"High Success" actor Harrison Ford was in the successful movie Star Wars*.  But in 1977, Harrison Ford was not highly successful yet.  The model needs to be retrained using more timely metrics of success.

## Files
[IMDB_movies.zip](https://github.com/krochkind/IMDB/blob/main/IMDB_movies.zip) - Original Data Set (20 mb zipped, 47 mb unzipped)<br />
[IMDB_movies_cleaned.csv](https://github.com/krochkind/IMDB/blob/main/IMDB_movies_cleaned.csv) - Cleaned Data Set<br />
[IMDB.ipynb](https://github.com/krochkind/IMDB/blob/main/IMDB.ipynb) - Jupyter Notebook with all code
