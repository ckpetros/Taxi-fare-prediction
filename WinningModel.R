
library(data.table)
library(Metrics)
library(Matrix)
library(xgboost) 

################
# Read in Data #
################
rm(list = ls())

data_path='/home/petro'
taxi <- fread(paste(data_path, "train_raw_and_new_features_3.csv", sep="/"))
test <- fread(paste(data_path, "test_raw_and_new_features_3.csv", sep="/"))

taxi <- taxi[, -c("INDEX", 
                  "tpep_pickup_datetime", 
                  "tpep_dropoff_datetime",
                  "pickup_neighborhood",
                  "dropoff_neighborhood"
                  
                  )]


test <- test[, -c( "id","tpep_pickup_datetime", 
                  "tpep_dropoff_datetime")]




## Reorder test so that it has the same order than train #
test$trip_duration <- (test$trip_duration / 60)


#test <- setcolorder(test, c('INDEX',colnames(taxi[, -"total_amount"])))

all(colnames(test[, - 'INDEX']) == colnames(taxi[, -"total_amount"])) || stop("Columns are not the same in test and train")



###################
# Prepare Factors #
###################

# neighborhood levels #
#neighborhoods_pickup_levels <- unique(c(unique(taxi$pickup_neighborhood), 
                                 unique(test$pickup_neighborhood)))

#neighborhoods_dropoff_levels <- unique(c(unique(taxi$dropoff_neighborhood),
                                  unique(test$dropoff_neighborhood)))


# cluster levels  #
#start_cluster_levels <- sort(unique(c(unique(taxi$Cluster_start), 
                               unique(test$Cluster_start))), 
                      decreasing = FALSE)

#end_cluster_levels <- sort(unique(c(unique(taxi$Cluster_end), 
                             unique(test$Cluster_end))), 
                    decreasing = FALSE)

#both_cluster_levels <- sort(unique(c(unique(taxi$Cluster_both), 
                              unique(test$Cluster_both))), 
                     decreasing = FALSE)

# Make all factors to factors #
#taxi$VendorID <- as.factor(taxi$VendorID)
#taxi$payment_type <- as.factor(taxi$payment_type)
taxi$store_and_fwd_flag <- as.numeric(as.factor(taxi$store_and_fwd_flag))
#taxi$pickup_neighborhood <- factor(taxi$pickup_neighborhood,
                                   levels = neighborhoods_pickup_levels)
#taxi$dropoff_neighborhood <- factor(taxi$dropoff_neighborhood,
                                    levels = neighborhoods_dropoff_levels)
#taxi$Cluster_both <- factor(taxi$Cluster_both,
                            levels = both_cluster_levels)
#taxi$Cluster_start <- factor(taxi$Cluster_start,
                             levels = start_cluster_levels)
#taxi$Cluster_end <- factor(taxi$Cluster_end,
                           levels = end_cluster_levels)


#test$VendorID <- as.factor(test$VendorID)
#test$payment_type <- as.factor(test$payment_type)
test$store_and_fwd_flag <- as.numeric(as.factor(test$store_and_fwd_flag))
#test$pickup_neighborhood <- factor(test$pickup_neighborhood,
                                   levels = neighborhoods_pickup_levels)
#test$dropoff_neighborhood <- factor(test$dropoff_neighborhood,
                                    levels = neighborhoods_dropoff_levels)
#test$Cluster_both <- factor(test$Cluster_both,
                            levels = both_cluster_levels)
#test$Cluster_start <- factor(test$Cluster_start,
                             levels = start_cluster_levels)
#test$Cluster_end <- factor(test$Cluster_end,
                           levels = end_cluster_levels)

gc(verbose = TRUE)


########################
# Create Sparse Matrix #
########################
taxi_sparse <- sparse.model.matrix(~., data = taxi)[,-1]
test_sparse <- sparse.model.matrix(~., data = test)[,-1]

colnames(taxi_sparse)
colnames(test_sparse)

######################
# Train / Test Split #
######################
set.seed(2018)
n <- nrow(taxi)
index <- sample(n, size =  0.8 * n, replace = FALSE)

taxi_train_sparse <- taxi_sparse[index,]
taxi_test_sparse <- taxi_sparse[-index,]


############################
# Split Models by Ratecode #
############################

# RatecodeID 1 #
taxi_train_sparse_rc1 <- taxi_train_sparse[taxi_train_sparse[, "RatecodeID"] == 1,]
taxi_test_sparse_rc1 <- taxi_test_sparse[taxi_test_sparse[, "RatecodeID"] == 1,]
test_sparse_rc1 <- test_sparse[test_sparse[, "RatecodeID"] == 1,]

# RatecodeID 2 #
taxi_train_sparse_rc2 <- taxi_train_sparse[taxi_train_sparse[, "RatecodeID"] == 2,]
taxi_test_sparse_rc2 <- taxi_test_sparse[taxi_test_sparse[, "RatecodeID"] == 2,]
test_sparse_rc2 <- test_sparse[test_sparse[, "RatecodeID"] == 2,]

# RatecodeID 3 #
taxi_train_sparse_rc3 <- taxi_train_sparse[taxi_train_sparse[, "RatecodeID"] == 3,]
taxi_test_sparse_rc3 <- taxi_test_sparse[taxi_test_sparse[, "RatecodeID"] == 3,]
test_sparse_rc3 <- test_sparse[test_sparse[, "RatecodeID"] == 3,]

# RatecodeID 4 #
taxi_train_sparse_rc4 <- taxi_train_sparse[taxi_train_sparse[, "RatecodeID"] == 4,]
taxi_test_sparse_rc4 <- taxi_test_sparse[taxi_test_sparse[, "RatecodeID"] == 4,]
test_sparse_rc4 <- test_sparse[test_sparse[, "RatecodeID"] == 4,]

# RatecodeID 5 #
taxi_train_sparse_rc5 <- taxi_train_sparse[taxi_train_sparse[, "RatecodeID"] == 5,]
taxi_test_sparse_rc5 <- taxi_test_sparse[taxi_test_sparse[, "RatecodeID"] == 5,]
test_sparse_rc5 <- test_sparse[test_sparse[, "RatecodeID"] == 5,]


rm(list = c("taxi", 'test', "taxi_train_sparse",  "taxi_test_sparse", "taxi_sparse"))
gc(verbose = TRUE)


colnames(taxi_train_sparse_rc1)
colnames(test_sparse_rc1[, -which(colnames(test_sparse_rc1) == "INDEX")])

################
# Fit Models #
################


# RatecodeID 1 #

set.seed(2018)

model_fit_rc1 <- xgboost(data = taxi_train_sparse_rc1[, -which(colnames(taxi_train_sparse_rc1) == "total_amount")],
                     label = taxi_train_sparse_rc1[, "total_amount"],
                     booster = "gbtree",
                     nthread = 8,
                     objective = "reg:linear",
                     eval_metric = "mae",
                     eta = 0.1,
                     gamma = 0,
                     max_depth = 10,
                     min_child_weight = 3,
                     subsample = 0.9, 
                     colsample_bytree = 0.9,
                     tree_method = "approx",
                     nrounds = 100,
                     early_stopping_rounds = 15) 

# evaluation:
model_pred_eval_rc1 <- predict(model_fit_rc1, taxi_test_sparse_rc1[, -which(colnames(taxi_train_sparse_rc1) == "total_amount")])
mae(round(model_pred_eval_rc1, digits = 2), taxi_test_sparse_rc1[, "total_amount"])



Index_rc1 = test_sparse_rc1[,"INDEX"]
# Make part of prediction #
model_pred_rc1 <- predict(model_fit_rc1, test_sparse_rc1[, -which(colnames(test_sparse_rc1) == "INDEX")])

submission_rc1 <- cbind(Index_rc1, model_pred_rc1)

gc(verbose = TRUE)
# RatecodeID 2 #


set.seed(2018)

model_fit_rc2 <- xgboost(data = taxi_train_sparse_rc2[, -which(colnames(taxi_train_sparse_rc2) == "total_amount")],
                         label = taxi_train_sparse_rc2[, "total_amount"],
                         booster = "gbtree",
                         nthread = 8,
                         objective = "reg:linear",
                         eval_metric = "mae",
                         eta = 0.05,
                         max_depth = 7,
                         min_child_weight = 1,
                         subsample = 1, 
                         colsample_bytree = 0.9,
                         tree_method = "approx",
                         nrounds = 250,
                         early_stopping_rounds = 25,
                         silent = 1,
                         verbose = 0) 

# evaluation:
model_pred_eval_rc2 <- predict(model_fit_rc2, taxi_test_sparse_rc2[, -which(colnames(taxi_train_sparse_rc2) == "total_amount")])

mae(round(model_pred_eval_rc2, digits = 2), taxi_test_sparse_rc2[, "total_amount"])



Index_rc2 = test_sparse_rc2[,"INDEX"]
# Make part of prediction #
model_pred_rc2 <- predict(model_fit_rc2, test_sparse_rc2[, -which(colnames(test_sparse_rc2) == "INDEX")])

submission_rc2 <- cbind(Index_rc2, model_pred_rc2)


gc(verbose = TRUE)
# RatecodeID 3 #

set.seed(2018)

model_fit_rc3 <- xgboost(data = taxi_train_sparse_rc3[, -which(colnames(taxi_train_sparse_rc3) == "total_amount")],
                         label = taxi_train_sparse_rc3[, "total_amount"],
                         booster = "gbtree",
                         nthread = 8,
                         objective = "reg:linear",
                         eval_metric = "mae",
                         eta = 0.009,
                         gamma = 0,
                         max_depth = 7,
                         min_child_weight = 4,
                         subsample = 0.9, 
                         colsample_bytree = 0.9,
                         tree_method = "approx",
                         nrounds = 1200,
                         early_stopping_rounds = 25,
                         verbose = 0) 

# evaluation:
model_pred_eval_rc3 <- predict(model_fit_rc3, taxi_test_sparse_rc3[, -which(colnames(taxi_train_sparse_rc3) == "total_amount")])
mae(round(model_pred_eval_rc3, digits = 2), taxi_test_sparse_rc3[, "total_amount"])





Index_rc3 = test_sparse_rc3[,"INDEX"]
# Make part of prediction #
model_pred_rc3 <- predict(model_fit_rc3, test_sparse_rc3[, -which(colnames(test_sparse_rc3) == "INDEX")])

submission_rc3 <- cbind(Index_rc3, model_pred_rc3)


gc(verbose = TRUE)

# RatecodeID 4 #
set.seed(2018)

model_fit_rc4 <- xgboost(data = taxi_train_sparse_rc4[, -which(colnames(taxi_train_sparse_rc4) == "total_amount")],
                         label = taxi_train_sparse_rc4[, "total_amount"],
                         booster = "gbtree",
                         nthread = 8,
                         objective = "reg:linear",
                         eval_metric = "mae",
                         eta = 0.1,
                         gamma = 0,
                         max_depth = 10,
                         min_child_weight = 3,
                         subsample = 0.9, 
                         colsample_bytree = 0.9,
                         tree_method = "approx",
                         nrounds = 100,
                         early_stopping_rounds = 15) 

# evaluation:
model_pred_eval_rc4 <- predict(model_fit_rc4, taxi_test_sparse_rc4[, -which(colnames(taxi_train_sparse_rc4) == "total_amount")])
mae(round(model_pred_eval_rc4, digits = 2), taxi_test_sparse_rc4[, "total_amount"])



Index_rc4 = test_sparse_rc4[,"INDEX"]
# Make part of prediction #
model_pred_rc4 <- predict(model_fit_rc4, test_sparse_rc4[, -which(colnames(test_sparse_rc4) == "INDEX")])

submission_rc4 <- cbind(Index_rc4, model_pred_rc4)

gc(verbose = TRUE)

# RatecodeID 5 #
model_fit_rc5 <- xgboost(data = taxi_train_sparse_rc5[, -which(colnames(taxi_train_sparse_rc5) == "total_amount")],
                         label = taxi_train_sparse_rc5[, "total_amount"],
                         booster = "gbtree",
                         nthread = 8,
                         objective = "reg:linear",
                         eval_metric = "mae",
                         eta = 0.1,
                         gamma = 0,
                         max_depth = 10,
                         min_child_weight = 3,
                         subsample = 0.9, 
                         colsample_bytree = 0.9,
                         tree_method = "approx",
                         nrounds = 100,
                         early_stopping_rounds = 15) 

# evaluation:
model_pred_eval_rc5 <- predict(model_fit_rc5, taxi_test_sparse_rc5[, -which(colnames(taxi_train_sparse_rc5) == "total_amount")])
mae(round(model_pred_eval_rc5, digits = 2), taxi_test_sparse_rc5[, "total_amount"])



Index_rc5 = test_sparse_rc5[,"INDEX"]
# Make part of prediction #
model_pred_rc5 <- predict(model_fit_rc5, test_sparse_rc5[, -which(colnames(test_sparse_rc5) == "INDEX")])

submission_rc5 <- cbind(Index_rc5, model_pred_rc5)



##############################
# Puzzle together submission #
##############################

submission <- rbind(submission_rc1, submission_rc2, submission_rc3, submission_rc4, submission_rc5)

submission <- submission[order(submission[,1]), ]

# evaluation:
ref_sub_1 <- fread("/home/petro/submission7_08_06_2018_16_36.csv")
mae(round(submission[,2] , digits = 2), round(ref_sub_1$total_amount, digits = 2))

ref_sub_2 <- fread("/home/petro/submission8_08_06_2018-jr.csv")
mae(round(submission[,2] , digits = 2), round(ref_sub_2$total_amount, digits = 2))

ref_sub_3 <- fread("/home/petro/submission9_09_06_2018-jr.csv")
mae(round(submission[,2] , digits = 2), round(ref_sub_3$total_amount, digits = 2))

colnames(ref_sub_1)
########################
# Write out submission #
########################
colnames(submission) <- c("ID", "total_amount")


submission[,"total_amount"] <- round(submission[,"total_amount"], digits = 2)



submission[submission[, "total_amount"] < 0 ,"total_amount"] <- 0

submission[,"ID"] <- 1:64000

write.table(submission, "/home/petro/submission14.csv", sep = ",", row.names = FALSE) 

