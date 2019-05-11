################################################################
##########################First Model##########################

# set working direction
setwd('/Users/Zoe/Desktop/Columbia University /Courses/APAN5200/Assignment/Kaggle Competition/Raw data')

# following code will read data
data = read.csv('analysisData.csv')
testing = read.csv('scoringData.csv')

# Explore the analysis dataset
# Understanding the structure of data
str(data)
# View its class
class(data)
# View its dimensions
dim(data)
# Look at column names
names(data)
# Visualize missing values of data
library(naniar)
gg_miss_var(data) # variables with missing value: "picture_url","thumbnail_url","medium_url","license","monthly_price","square_feet","weekly_price","security_deposit","cleaning_fee","beds"
# If any did have near zero variance, then we would consider removing that feature
nearZeroVar(data[, -61], saveMetrics = TRUE) 

# Remove variables with zero variance and high propotion of missing value
variables_to_romove1 <- c('state','market','country','scrape_id','experiences_offered','thumbnail_url','medium_url','xl_picture_url','host_acceptance_rate','has_availability','requires_license','license',"monthly_price","square_feet","weekly_price","require_guest_phone_verification","require_guest_profile_picture","host_has_profile_pic","country_code")
data <- data[,!colnames(data) %in% variables_to_romove1, drop=F]
# Remove variables with too many levels and no contribution to prediction
variables_to_romove2 <- c("smart_location","street","host_location","listing_url","last_scraped","name","summary","space","description","neighborhood_overview","notes","transit","access","interaction","house_rules","picture_url","host_id","host_url","host_name","host_since","host_location","host_about","host_thumbnail_url","host_picture_url","host_neighbourhood","host_verifications","amenities","calendar_updated","first_review","last_review","city")
data <- data[,!colnames(data) %in% variables_to_romove2, drop=F]
# Remove id since it should have no value in prediction
data$id = NULL

# Convert character columns to factor, filling NA values with "missing"
for (col in colnames(data)){
  if (typeof(data[,col]) == "character"){
    new_col = data[,col]
    new_col[is.na(new_col)] = "missing"
    data[col] = as.factor(new_col)
  }
}

# Fill remaining NA values with -1
data[is.na(data)] = -1

# whether any variables have a correlation with price with an absolute value of 0.5 or higher
for (col in colnames(data)){
  if(is.numeric(data[,col])){
    if( abs(cor(data[,col],data$price)) > 0.5){
      print(col)
      print( cor(data[,col],data$price) )
    }
  }
} #"accommodates" and "cleaning_fee"

# which numeric variables have low correlations with sales prices
for (col in colnames(data)){
  if(is.numeric(data[,col])){
    if( abs(cor(data[,col],data$price)) < 0.1){
      print(col)
      print( cor(data[,col],data$price) )
    }
  }
} #"host_listings_count","host_total_listings_count","minimum_nights","maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_value","calculated_host_listings_count","reviews_per_month"

# Determine whether any of the numeric variables are highly correlated with one another
# Correlations of numeric data to check for bivariate correlations
nums <- sapply(data, is.numeric)
cor(data[,nums]) #a cutoff that I usually use is +-0.7
# host_listings_count and host_total_listings_count,beds and accomendates,availability_n,review_scores_rating and review_scores_accuracy,review_scores_rating and review_scores_cleanliness,review_scores_value and review_scores_rating
# Since we will be using a tree-based model, we deal with variables with correlations above 0.9 only 
# Remove host_listings_count,"availability_30","availability_60","availability_90"
variables_to_romove3 <- c("host_listings_count","availability_30","availability_60","availability_90")
data <- data[,!colnames(data) %in% variables_to_romove3, drop=F]

# Decorate cancellation_policy
## new cancellation_policy <- c("flexible", "moderate", "strict")
data <-data %>%
  mutate(cancellation_policy = case_when(cancellation_policy == "flexible" ~"flexible",
                                         cancellation_policy == "moderate" ~"moderate",
                                         TRUE ~ "strict"))
# Visualize correlation between cancellation_policy and price
ggplot(data,aes(x=cancellation_policy,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate property_type
## new property_type <- c("Apartment", "House", "Condo", "Loft")
data <-data %>%
  mutate(property_type = case_when(property_type == "Apartment" ~"Apt",
                                   property_type == "House" ~ "House",
                                   property_type == "Condominium" ~ "Condo",
                                   property_type == "Loft"  ~ "Loft",
                                   TRUE ~ "Rest"))
# Visualize correlation between property type and price
ggplot(data,aes(x=property_type,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate bed_type
## new bed_type <- c("Real Bed", "Not Real Bed")
data <-data %>%
  mutate(bed_type  = case_when(bed_type == "Real Bed" ~"Real Bed",
                               TRUE ~ "Not Real Bed"))
# Visualize correlation between bed type and price
ggplot(data,aes(x=bed_type,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate room_type
## room_type <- c("Entire home","Room")
data <-data %>%
  mutate(room_type = case_when(room_type == "Entire home/apt" ~"Entire home",
                               TRUE ~ "Room"))
# Visualize correlation between room type and price
ggplot(data,aes(x=room_type,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate zipcode
## creat new variable based on zipcode: borough <- c("Bronx","Brooklyn","Manhattan","Queens","StatenIsland","NA")
class(data$zipcode) #"factor"
as.numeric(data$zipcode) #convert zipcode to numeric
borough_Bronx <- c(10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, 10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473)
borough_Brooklyn <- c(11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, 11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, 11237)
borough_Manhattan <- c(10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040)
borough_Queens <- c(11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11368, 11369, 11370, 11372, 11373, 11377, 11378)
borough_StatenIsland <- c(10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312, 10301, 10304, 10305, 10314)
borough <- c(borough_Bronx,borough_Brooklyn,borough_Manhattan,borough_Queens,borough_StatenIsland)
data <-data %>%
  mutate(borough = case_when(zipcode %in% borough_Bronx ~"Bronx",
                             zipcode %in% borough_Brooklyn ~"Brooklyn",
                             zipcode %in% borough_Manhattan ~"Manhattan",
                             zipcode %in% borough_Queens ~"Queens",
                             zipcode %in% borough_StatenIsland ~"StatenIsland",
                             TRUE ~ "NA" ))

# Remove variables because of different levels
variables_to_romove4 <- c("neighbourhood","neighbourhood_cleansed","neighbourhood_group_cleansed","jurisdiction_names","zipcode")
data <- data[,!colnames(data) %in% variables_to_romove4, drop=F]

# Remove record of Uruguay which is not useful for a model to predict house price in NY
data <- data[-4568,]

# Create a predictive model
# Loading in some pacakges
library(caret)
library(plyr)
library(xgboost)
library(Metrics)
# Create custom summary function in proper format for caret
custom_summary = function(data, lev = NULL, model = NULL){
  out = rmsle(data[, "obs"], data[, "pred"])
  names(out) = c("rmsle")
  out
}
# Create control object
control = trainControl(method = "cv",  # Use cross validation
                       number = 10,     # 10-folds
                       summaryFunction = custom_summary                      
)
# Create grid of tuning parameters
grid = expand.grid(nrounds=c(100, 200, 400, 800), # Test 4 values for boosting rounds
                   max_depth= c(4, 6),           # Test 2 values for tree depth
                   eta=c(0.1, 0.05, 0.025),      # Test 3 values for learning rate
                   gamma= c(0.1,0.5), 
                   colsample_bytree = c(1), 
                   min_child_weight = c(1),
                   subsample=0.5)
# Train the model, using the custom metric rmsle
set.seed(12)
xgb_tree_model =  train(price~.+ number_of_reviews*review_scores_rating+room_type*property_type + beds*bed_type + is_location_exact*borough,      # Predict SalePrice using all features plus two interactions: room_type and property_type, beds and bed_type, is_location_exact and borough
                        data=data,
                        method="xgbTree",
                        trControl=control, 
                        tuneGrid=grid, 
                        metric="rmsle",     # Use custom performance metric
                        maximize = FALSE)   # Minimize the metric
# Check the results of training and which tuning parameters were selected
xgb_tree_model$results
xgb_tree_model$bestTune
# Check which variables ended up being most important to the model
varImp(xgb_tree_model)

# Make predictions on the data set using the model
data_predictions = predict(xgb_tree_model, data)

# Evaluation of model
sse=sum((data_predictions-data$price)^2)
sst=sum((mean(data$price)-data$price)^2)
rmse=sqrt(mean((data_predictions-data$price)^2)) 
rmse

# Deal with testing data
# Decorate property_type
testing <-testing %>%
  mutate(property_type = case_when(property_type == "Apartment" ~"Apt",
                                   property_type == "House" ~ "House",
                                   property_type == "Condominium" ~ "Condo",
                                   property_type == "Loft"  ~ "Loft",
                                   TRUE ~ "Rest"))

# Decorate bed_type
testing <-testing %>%
  mutate(bed_type = case_when(bed_type == "Real Bed" ~"Real Bed",
                              TRUE ~ "Not Real Bed"))

# Decorate room_type
testing <-testing %>%
  mutate(room_type = case_when(room_type == "Entire home/apt" ~"Entire home",
                               TRUE ~ "Room"))

# Decorate cancellation_policy
testing <-testing %>%
  mutate(cancellation_policy = case_when(cancellation_policy == "flexible" ~"flexible",
                                         cancellation_policy == "moderate" ~"moderate",
                                         TRUE ~ "strict"))

# Decorate zipcode
## creat new variable based on zipcode: borough <- c("Bronx","Brooklyn","Manhattan","Queens","StatenIsland","NA")
class(testing$zipcode) #"factor"
as.numeric(testing$zipcode) #convert zipcode to numeric
borough_Bronx <- c(10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, 10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473)
borough_Brooklyn <- c(11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, 11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, 11237)
borough_Manhattan <- c(10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040)
borough_Queens <- c(11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11368, 11369, 11370, 11372, 11373, 11377, 11378)
borough_StatenIsland <- c(10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312, 10301, 10304, 10305, 10314)
testing <-testing %>%
  mutate(borough = case_when(zipcode %in% borough_Bronx ~"Bronx",
                             zipcode %in% borough_Brooklyn ~"Brooklyn",
                             zipcode %in% borough_Manhattan ~"Manhattan",
                             zipcode %in% borough_Queens ~"Queens",
                             zipcode %in% borough_StatenIsland ~"StatenIsland",
                             TRUE ~ "NA"))

# Replace all NA values with -1
testing[is.na(testing)]<--1

# read in scoring data and apply model to generate predictions scoringData
test_predictions_1 = predict(xgb_tree_model, newdata=testing)

###############################################################
##########################Second Model##########################

# set working direction
setwd('/Users/Zoe/Desktop/Columbia University /Courses/APAN5200/Assignment/Kaggle Competition/Raw data')

# following code will read data
data = read.csv('analysisData.csv')
testing = read.csv('scoringData.csv')

# Explore the analysis dataset
# Understanding the structure of data
str(data)
# View its class
class(data)
# View its dimensions
dim(data)
# Look at column names
names(data)
# Visualize missing values of data
library(naniar)
gg_miss_var(data) # variables with missing value: "picture_url","thumbnail_url","medium_url","license","monthly_price","square_feet","weekly_price","security_deposit","cleaning_fee","beds"
# If any did have near zero variance, then we would consider removing that feature
nearZeroVar(data[, -61], saveMetrics = TRUE) 

# Remove variables with zero variance and high propotion of missing value
variables_to_romove1 <- c('state','market','country','scrape_id','experiences_offered','thumbnail_url','medium_url','xl_picture_url','host_acceptance_rate','has_availability','requires_license','license',"monthly_price","square_feet","weekly_price","require_guest_phone_verification","require_guest_profile_picture","host_has_profile_pic","country_code")
data <- data[,!colnames(data) %in% variables_to_romove1, drop=F]
# Remove variables with too many levels and no contribution to prediction
variables_to_romove2 <- c("smart_location","street","host_location","listing_url","last_scraped","name","summary","space","description","neighborhood_overview","notes","transit","access","interaction","house_rules","picture_url","host_id","host_url","host_name","host_since","host_location","host_about","host_thumbnail_url","host_picture_url","host_neighbourhood","host_verifications","amenities","calendar_updated","first_review","last_review","city")
data <- data[,!colnames(data) %in% variables_to_romove2, drop=F]
# Remove id since it should have no value in prediction
data$id = NULL

# Convert character columns to factor, filling NA values with "missing"
for (col in colnames(data)){
  if (typeof(data[,col]) == "character"){
    new_col = data[,col]
    new_col[is.na(new_col)] = "missing"
    data[col] = as.factor(new_col)
  }
}

# Fill remaining NA values with -1
data[is.na(data)] = -1

# whether any variables have a correlation with price with an absolute value of 0.5 or higher
for (col in colnames(data)){
  if(is.numeric(data[,col])){
    if( abs(cor(data[,col],data$price)) > 0.5){
      print(col)
      print( cor(data[,col],data$price) )
    }
  }
} #"accommodates" and "cleaning_fee"

# which numeric variables have low correlations with sales prices
for (col in colnames(data)){
  if(is.numeric(data[,col])){
    if( abs(cor(data[,col],data$price)) < 0.1){
      print(col)
      print( cor(data[,col],data$price) )
    }
  }
} #"host_listings_count","host_total_listings_count","minimum_nights","maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_value","calculated_host_listings_count","reviews_per_month"

# Determine whether any of the numeric variables are highly correlated with one another
# Correlations of numeric data to check for bivariate correlations
nums <- sapply(data, is.numeric)
cor(data[,nums]) #a cutoff that I usually use is +-0.7
# host_listings_count and host_total_listings_count,beds and accomendates,availability_n,review_scores_rating and review_scores_accuracy,review_scores_rating and review_scores_cleanliness,review_scores_value and review_scores_rating
# Since we will be using a tree-based model, we deal with variables with correlations above 0.9 only 
# Remove host_listings_count,"availability_30","availability_60","availability_90"
variables_to_romove3 <- c("host_listings_count","availability_30","availability_60","availability_90")
data <- data[,!colnames(data) %in% variables_to_romove3, drop=F]

# Decorate cancellation_policy
## new cancellation_policy <- c("flexible", "moderate", "strict")
data <-data %>%
  mutate(cancellation_policy = case_when(cancellation_policy == "flexible" ~"flexible",
                                         cancellation_policy == "moderate" ~"moderate",
                                         TRUE ~ "strict"))
# Visualize correlation between cancellation_policy and price
ggplot(data,aes(x=cancellation_policy,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate property_type
## new property_type <- c("Apartment", "House", "Condo", "Loft")
data <-data %>%
  mutate(property_type = case_when(property_type == "Apartment" ~"Apt",
                                   property_type == "House" ~ "House",
                                   property_type == "Condominium" ~ "Condo",
                                   property_type == "Loft"  ~ "Loft",
                                   TRUE ~ "Rest"))
# Visualize correlation between property type and price
ggplot(data,aes(x=property_type,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate bed_type
## new bed_type <- c("Real Bed", "Not Real Bed")
data <-data %>%
  mutate(bed_type  = case_when(bed_type == "Real Bed" ~"Real Bed",
                               TRUE ~ "Not Real Bed"))
# Visualize correlation between bed type and price
ggplot(data,aes(x=bed_type,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate room_type
## room_type <- c("Entire home","Room")
data <-data %>%
  mutate(room_type = case_when(room_type == "Entire home/apt" ~"Entire home",
                               TRUE ~ "Room"))
# Visualize correlation between room type and price
ggplot(data,aes(x=room_type,y=price)) +
  stat_summary(fun.y=mean, geom='bar')

# Decorate zipcode
## creat new variable based on zipcode: borough <- c("Bronx","Brooklyn","Manhattan","Queens","StatenIsland","NA")
class(data$zipcode) #"factor"
as.numeric(data$zipcode) #convert zipcode to numeric
borough_Bronx <- c(10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, 10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473)
borough_Brooklyn <- c(11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, 11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, 11237)
borough_Manhattan <- c(10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040)
borough_Queens <- c(11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11368, 11369, 11370, 11372, 11373, 11377, 11378)
borough_StatenIsland <- c(10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312, 10301, 10304, 10305, 10314)
borough <- c(borough_Bronx,borough_Brooklyn,borough_Manhattan,borough_Queens,borough_StatenIsland)
data <-data %>%
  mutate(borough = case_when(zipcode %in% borough_Bronx ~"Bronx",
                             zipcode %in% borough_Brooklyn ~"Brooklyn",
                             zipcode %in% borough_Manhattan ~"Manhattan",
                             zipcode %in% borough_Queens ~"Queens",
                             zipcode %in% borough_StatenIsland ~"StatenIsland",
                             TRUE ~ "NA" ))

# Remove variables because of different levels
variables_to_romove4 <- c("neighbourhood","neighbourhood_cleansed","neighbourhood_group_cleansed","jurisdiction_names","zipcode")
data <- data[,!colnames(data) %in% variables_to_romove4, drop=F]

# Remove record of Uruguay which is not useful for a model to predict house price in NY
data <- data[-4568,]

# Create a predictive model
# Loading in some pacakges
library(caret)
library(plyr)
library(xgboost)
library(Metrics)
set.seed(100)
trControl=trainControl(method="cv",number=10)
tuneGrid=  expand.grid(n.trees = 1000, interaction.depth = c(1,2),
                       shrinkage = (1:100)*0.001,n.minobsinnode=5)
cvBoost = train(price~.+ number_of_reviews*review_scores_rating+room_type*property_type + beds*bed_type + is_location_exact*borough,      # Predict SalePrice using all features plus two interactions: room_type and property_type, beds and bed_type, is_location_exact and borough
                data=data,
                method="gbm",
                trControl=trControl,
                tuneGrid=tuneGrid)

# Make predictions on the data set using the model
data_predictions = predict(cvBoost, data)

# Evaluation of model
sse=sum((data_predictions-data$price)^2)
sst=sum((mean(data$price)-data$price)^2)
rmse=sqrt(mean((data_predictions-data$price)^2)) 
rmse

# Deal with testing data
# Decorate property_type
testing <-testing %>%
  mutate(property_type = case_when(property_type == "Apartment" ~"Apt",
                                   property_type == "House" ~ "House",
                                   property_type == "Condominium" ~ "Condo",
                                   property_type == "Loft"  ~ "Loft",
                                   TRUE ~ "Rest"))

# Decorate bed_type
testing <-testing %>%
  mutate(bed_type = case_when(bed_type == "Real Bed" ~"Real Bed",
                              TRUE ~ "Not Real Bed"))

# Decorate room_type
testing <-testing %>%
  mutate(room_type = case_when(room_type == "Entire home/apt" ~"Entire home",
                               TRUE ~ "Room"))

# Decorate cancellation_policy
testing <-testing %>%
  mutate(cancellation_policy = case_when(cancellation_policy == "flexible" ~"flexible",
                                         cancellation_policy == "moderate" ~"moderate",
                                         TRUE ~ "strict"))

# Decorate zipcode
## creat new variable based on zipcode: borough <- c("Bronx","Brooklyn","Manhattan","Queens","StatenIsland","NA")
class(testing$zipcode) #"factor"
as.numeric(testing$zipcode) #convert zipcode to numeric
borough_Bronx <- c(10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, 10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473)
borough_Brooklyn <- c(11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, 11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, 11237)
borough_Manhattan <- c(10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040)
borough_Queens <- c(11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11368, 11369, 11370, 11372, 11373, 11377, 11378)
borough_StatenIsland <- c(10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312, 10301, 10304, 10305, 10314)
testing <-testing %>%
  mutate(borough = case_when(zipcode %in% borough_Bronx ~"Bronx",
                             zipcode %in% borough_Brooklyn ~"Brooklyn",
                             zipcode %in% borough_Manhattan ~"Manhattan",
                             zipcode %in% borough_Queens ~"Queens",
                             zipcode %in% borough_StatenIsland ~"StatenIsland",
                             TRUE ~ "NA"))

# Replace all NA values with -1
testing[is.na(testing)]<--1

# read in scoring data and apply model to generate predictions scoringData
test_predictions_2 = predict(cvBoost, newdata=testing)

##################################################################
#################Combine two prediction together#################
test_predictions = test_predictions_1*0.75 + test_predictions_2*0.25

# construct submision from predictions
submissionFile = data.frame(id = testing$id, price = test_predictions)
write.csv(submissionFile, 'sample_submission.csv',row.names = F)
