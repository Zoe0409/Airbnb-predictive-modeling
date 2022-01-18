# Airbnb-predictive-modeling

## Description

The dataset contains listings of over 25,000 Airbnb rentals in New York City. The goal of the Airbnb Price prediction competition is to predict the price for a rental using over 90 variables on the property, host, and past reviews.

## Metric

Submissions will be evaluated based on RMSE (root mean squared error). Lower the RMSE, better the model.

This competition aims at predicting prices of Airbnb rooms in New York based on a large set of features. It starts by acquiring and defining input data, data cleaning, feature engineering, and finally by model training, selection, and prediction. The research predicts house prices by combining results of several Machine Learning algorithms. I divided my whole process into four steps:

 - Assemble the data and explore it

 - Data cleansing, build what is needed
 
 - Model selection

 - Final prediction

Several R packages are used in the research. Packages cover data visualization, data manipulation, and model training and prediction.

### Assemble the data and explore it


When I went through the whole analysis dataset, I found that there is one record from Uruguay and that there are 25 houses the price of which is zero. Since the prediction is based on houses in New York and it is not reasonable to believe that houses could be free, I removed these 26 records altogether.

In order to operate on the train dataset and test dataset in the same way and to take these data into consideration when exploring variables, I combine train dataset and test dataset and name it as fulldata after removing the target variable from train dataset and naming it as SalePrice list.

### Data cleansing, build what is needed

Before cleaning the data, I took a lot at the variable list to see what we have here by using str(fulldata), class(fulldata), dim(fulldata) and names(fulldata). By exploring the data set, I realized that the data needs to be processed and cleaned. Data cleaning involves handling missing values, fixing features classes, or coding character features.

Impute missing value: Knowing the number of missing values and taking a closer look at the data description, we can see that NA does not always mean that the values are missing. 

For the features "beds", "cleaning_fee", NA just means that there is "no beds"/"no cleaning_fee" in the room. Since I would use tree models, I just coded such NA values to be -1.

In case of the features "xl_picture_url", "thumbnail_url", "medium_url", "license", "security_deoposit”, there is only missing value so I just remove these features.

For “zipcode”, I removed this variable because I would use another variable to represent location information.

For those features with high proportion of missing value, I use whether there is a value or not to differentiate them by coding such missing value to be 0, and other values to be 1.

### Explore dataset variables

Exploring dataset could be difficult when the quantity of variables is quite huge. Therefore, I first explored variables from some categories and in next step, I decorated some variables one by one. The descriptive analysis of dummy variables was mostly conducted by drawing box plots. Some variables appeared to be ineffective due to the extreme box plot. The numeric variables are sorted out before turning dummy variables into numeric form.

Variance of each variable: I took a look at the variance of each variable by using nearZeroVar(), and I removed those variables with zero variance.

Categorical variables about location: There are a lot of categorical variables describing about location but they have two common issues: inconsistency and typos. But I found one variable "neighbourhood_group_cleansed" really clean. I double checked with zipcode after mutating a new variable “borough” based on the value of zipcode and found that “neighbourhood_group_cleansed” is consistent with “borough” so I just removed all other categorical variables about location.
Host information: There is no missing value in variables about host information and I don’t think host personal information would have an impact on price so I removed all these variables but “host_about” to test whether the length of description of host would have an impact on price later.

URL variables: Likely, for the URL variables without missing value, I didn’t decorate them but just removed them because I think they cannot contribute much to the prediction of price.

Self-related variables: Determine whether any of the numeric variables are highly correlated with one another. A cutoff that I usually use is +-0.7, but since we will be using a tree-based model, we deal with variables with correlations above 0.9 only. So I removed "availability_30","availability_60","availability_90" after checking for bivariate correlations of numeric data.
Date-related variables: I removed date-related variables with too many levels and little contribution for the prediction of price. But I left first_review and last_review because I want to mutate a new variable to see whether the gap between them is greater than one year would have an impact on price.

first_review and last_review: I mutated a new variable to calculate the gap between first_review and last_review, and I grouped them into two categories based on whether the gap is greater than 365 days or not.

### Data preparation

There are some variables that cannot be used directly but they have potential value and interesting stories to be explored, so I decorated these variables to new ones.

Calendar_updated: I coded this categorical variable with too many different levels to be numeric by counting how many days since calendar updated last time.

Descriptive variables: including Name, Access, Summary, Space, Description, Neighborhood, Interaction, Transit, Notes, House Rules, host_about. This process is really enjoyable, since you can find a lot of fun facts after text mining. Basically, there are three steps for each of these variables. First, I set three functions to find out the most frequent words used in that variable, check the existence of most frequent words, and measure the length of description respectively. I don’t think that adding one more word would have a significant impact on price so I just put them into two groups based on whether it is longer than the average length.

Amenities: I believe that the number of amenities matters a lot for price and I also noticed that the amenities with “” are advanced while the others are just basic. But I still cannot figure out how to count the number of amenities with or without “” separately so I just counted the number of amenities and put them into two groups based on whether it is more than the average.

property_type: this variable consists of a lot of levels so I selected four property types with the most observations and put the others into one same group altogether.

Review scores: I used linear regression to check the relationship among the variables about rating and found that the overall review score is determined by the other sub review scores from different dimensions. So I removed all these sub review scores and just remained the overall review score. 

host_response_rate: this variable seems to be numeric at first but actually it is factor. So I grouped them into four categories, including “N/A”, “Bad” for 0% to 20%, “Excellent” for 80% to 100%, and “Rest” for all the rest. 

### Split combined full dataset into train and test datasets

Before splitting, I converted character columns to factor. I split combined full dataset into train and test datasets based on the number of rows of “data” dataset and “testing” dataset. SalePrice was added back to train dataset and id was removed from train dataset.

### Further operation on variables left now

Track outliers: Using ggplot to see the correlation between some numeric variables and price, I found that for bathrooms, bedrooms, beds, and cleaning_fee, some outliers existed and twisted the relationship. After analyzing, I put the average value of each variable for these outliers.

Numeric variables: After checking the correlation of numeric variables with SalePrice, I removed those variables with correlation lower than 0.1

Categorical variables: Using visualized bar chart, I visualized the relationship between each categorical independent variable and SalePrice. I removed those categorical variables with no obvious difference for different categories.

Transform those values with high skewness-: After determining skew of each numeric variable and a threshold skewness 0.75, I transformed excessively skewed features with log(x+1)

### Model selection

Before building a model, I used the caret partitioning function to split train dataset into train_inner and test_inner to get rid of overfitting issue.

Linear Model: Since our outcome is a continuous numeric variable, we want a linear model, not a GLM. First, I just tossed it all in there and used a proper regression model as my first examination of the data, to get a feel for what’s there. Also, I applied the model to test_inner to predict SalePrice on test_inner dataset, Comparing the predicted price and actual price, I can know the RMSE on test_inner dataset

Advanced Tree Model: After setting seed and parameters for each model one by one, I trained different advanced tree model combined with the use of cross-validation on train_inner dataset. Later I put testing prediction and test dataset all together and tested with RMSE.

### Final prediction

Finally, I predicted using the test set and output file as instructions showed.

## Lessons from this competition

As a beginner of machine learning, I took this competition as a good opportunity for me to apply my knowledge into practice and also tried to learn more new knowledge in this field. Here are some lessons that I learnt during this process:
I got my best result in public leaderboard at the beginning of the competition when I didn't know how to play with text and just removed all the variables that I didn’t know how to deal with. As I dug deeper on some variables, such as the length of summary, the existence of some key words in name, and information about review date, I tried to find out the value behind them but somehow I cannot improve my result.

There are different limitations for different models but only after receiving errors or warning messages can I realize what I did wrong or what I missed when I built the models. 

Sometimes the models can be really time-consuming with cross-validation and the large amount of data to train. In order to save time, I can just use some sample of the whole dataset to find the best parameters and tune models.

There are some observations that don’t make sense but I still don’t know how to deal with them. For example, I found that except the 25 observations with price equaling to zero, there are some observations with price equaling to 999.
For the correlation between each categorical variable and price, I only use ggplot to visualize and make judgement by myself which can lead to mistakes.

There are still some variables need to be investigated. For example, how to differentiate basic amenities and advanced ones by “” because advanced amenities have a significant influence on price. For host_is_superhost, based on my experience as an Airbnb host, I know there are some requirements to be qualified as a superhost, like host_response_rate(>=90%), review_scores_rating(>=4.8), active(== 10+), cancellations(== 0). I wonder whether the other variables can be summarized and substituted by variable “host_is_superhost”.

I know feature selection is really important but I did not work much on it this time because I think that advanced trees can select important features automatically. However, overfitting became an issue with noise in model so next time, I should take some strategy on this part.

If you ask Airbnb room seekers to describe how much they are willing to pay for a room, they probably would not care about the personal information about host, but they would attach great importance to the location and accommodates. This Kaggle competition dataset proves that there are many more room features that influence price negotiations than the number of bedrooms. In order to improve performance next time, I should work on data preparation, feature engineering and model selection, using what I learnt from this competition.
