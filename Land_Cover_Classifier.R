#################################################################
# Title: "Supervised Classification of Sentinel-2 Satellite Image
#     for producing Land Use /Land Cover Map for the Gaza Strip"
# author: "Mohammed Alhessi"
# date: "1/2/2022"
#################################################################


############### IMPORTANT NOTE ##################################
# Please skip the steps from Step 1 to Step 4, and start from 
# Step 5. The Steps 1 to 4 are processing the satellite image, extracting the pixel values into R data frame, and save them in data/data.rda file, which was used in building the models.

#################################################################

#================================================================
# Install and Import Necessary Libraries
#================================================================

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(raster)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(modeest)) install.packages("modeest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(raster)
library(GGally)
library(MASS)
library(naivebayes)
library(nnet)
library(rpart)
library(randomForest)
library(kernlab)
library(modeest)

# Introduction

#================================================================
## Step 1: Download the Sentinel-2 Satellite Image
## Step 2: Stack and Crop the image 
#================================================================

# List all files in the image folder
imgFolder <- list.files("./data/S2B/S2B_MS~1.SAF/GRANULE/L2A_T3~1/IMG_DATA/R10m")

#Selection of desired bands (band 2, band3, band 4, and band 8)
bands <- c(grep('B02_10m', imgFolder, value=TRUE), # Blue band
           grep('B03_10m', imgFolder, value=TRUE), # Green band
           grep('B04_10m', imgFolder, value=TRUE), # Red band
           grep('B08_10m', imgFolder, value=TRUE)) # Near Infra-red band

# Create the Raster (image) Stack
rasterStack <- stack(paste0("./data/S2B/S2B_MS~1.SAF/GRANULE/L2A_T3~1/IMG_DATA/R10m/",bands))

# Write the Raster Stack to new file
writeRaster(rasterStack,paste0("./data/StackedRasterS2B",".tif"),
            format="GTiff", overwrite = TRUE)


# Load the stacked image/raster file
#The raster package offers '
#layer() - one file, one band
#stack() - multiple files, multiple bands
#brick() - one file, multiple bands

imgBrick <- brick("./data/StackedRasterS2B.tif")

# Plot and visualize the true color composite image of the visible bands (red, green, and blue bands).
plotRGB(imgBrick,r=3,g=2,b=1,stretch="lin")

# Read the Shapefile of the Gaza Strip (crop boundary)
gazaShp <- shapefile("data/gaza/gazaStrip.shp")

# Mask the image to the crop boundary. We don't need any pixels outside the boundary.
masked <- mask(x=imgBrick, mask=gazaShp)

# Crop the masked image
cropped <- crop(x=masked, y=gazaShp)

# Plot the cropped image
plotRGB(cropped,r=3,g=2,b=1,stretch="lin")

# Write the Cropped Raster Brick to new file
writeRaster(cropped,paste0("./data/CroppedRasterS2B",".tif"),
            format="GTiff", overwrite = TRUE)

#================================================================
## Step 3: Create Training Sites/Samples
## Step 4: Build the samples (pixel values) matrix
#================================================================#
# Read the shapefile of the training samples/polygons using shapefile() function from raster package.
SamplesShp <- shapefile("data/LCLU_Classes_65K/LULC_Classes_65K2.shp", stringsAsFactors=TRUE)

# This is to check that both image and shapefile have the same Coordinate Reference System (CRS) so that they geospatially align together and the training samples are located exactly in their positions on the image.
compareCRS(cropped,SamplesShp)

#Rename Image Bands
names(cropped) <- c("B2blue", "B3green", "B4red", "B8nearInraRed")

# Extract the pixel values from the cropped sentinel-2 image. The pixel values are extracted without thier corresponding labels.
samples_data <- raster::extract(cropped, SamplesShp,df=TRUE)

# We need to join the extracted pixel values with their corresponding labels from the sahpefile.
# Add n column to  the attribute table of shapefile to act like ID column that will be used for joining.
SamplesShpAttributes <- SamplesShp@data %>% mutate(n=1:n())

# Join the data from the sahpefile to extracted pixel values.
samples_data <- samples_data %>% left_join(SamplesShpAttributes, by=c("ID"="n"))

# Select the necessary columns.
data <- samples_data %>% dplyr::select(c(ID, B2blue, B3green, B4red, B8nearInraRed, Classname, Classvalue))

# Save the data for future use.
save(data, file="data/data.rda")

#===============================================================#
# Step 5: Explore, Clean and visualize the samples matrix
#===============================================================# 
# Load the saved data
load(file="data/data.rda")

# Explore the samples matrix
summary(data)

# Remove NAs Values
data <- data %>% filter(!is.na(B2blue))
knitr::kable(head(data), caption = "First rows of the data (samples matrix).")

### Plot the distribution of the classes
data %>% ggplot(aes(x=Classname, fill=Classname)) + geom_bar() +geom_text(aes(label=..count..),stat="count")

### Plot the spectral profile curves
# Calculate the mean pixel values by class and band
class_profiles <- data %>% dplyr::select(-ID) %>% group_by(Classname, Classvalue) %>% summarize_all(list(mean=mean)) %>% ungroup()

# Put the class_profiles in long format
class_profiles2 <- class_profiles %>% gather(key="band", value="mean_pixel_value", B2blue_mean:B8nearInraRed_mean) %>% mutate(band=str_remove(band,"_mean"))

# Plot the spectral profile
class_profiles2 %>% ggplot(aes(x=band, y=mean_pixel_value, group=Classname, color=Classname))+ geom_line() + geom_point()

### Explore the correlation between the bands
bands_corr_matrix <- data %>% dplyr::select(B2blue:B8nearInraRed) %>% cor()
knitr::kable(bands_corr_matrix, caption = "Correlation Matrix")

### Visualize the scatterplot matrix of bands and classes
ggpairs(data, columns = 2:5, ggplot2::aes(colour=Classname))

#================================================================
## Step 6: Partition the data into training, testing, and validation sets
#================================================================

# Partition the whole Data (63,801 Pixels) into training, test, and validation data
# Create the validation set of the whole data
set.seed(1, sample.kind="Rounding")
validation_index <- createDataPartition(y = data$Classvalue, times = 1, p = 0.1, list = FALSE)
train_samples_63k <- data[-validation_index,]
test_samples_63k <- data[validation_index,]

# Sample 10,000 Observations(Pixels) from the whole data for training purposes.
# To avoid imbalanced dataset, we need to sample 2500 pixel from each class.

# Sample 10,000 pixel values (samples). 
set.seed(3, sample.kind="Rounding")
builtup_sample <- slice_sample(data[data$Classvalue=="1",],n=2500,replace = TRUE)
agricultural_sample <- slice_sample(data[data$Classvalue=="2",],n=2500,replace = TRUE)
greenhouse_sample <- slice_sample(data[data$Classvalue=="3",],n=2500,replace = TRUE)
barren_sample <- slice_sample(data[data$Classvalue=="4",],n=2500,replace = TRUE)

data_10k <- bind_rows(builtup_sample,agricultural_sample,greenhouse_sample,barren_sample)

# Partition subset of Data (10,000 Pixels) into training, test, and validation data

# Create the validation set from the 10,000 samples
set.seed(4, sample.kind="Rounding")
validation_indx <- createDataPartition(y = data_10k$Classvalue, times = 1, p = 0.1, list = FALSE)
train_samples_10k <- data_10k[-validation_indx,]
test_samples_10k <- data_10k[validation_indx,]

#================================================================
## Step 7: Building the classification models  
#================================================================

### Quadratic Discriminant Analysis (QDA)
#-----------------------------------------

set.seed(10, sample.kind="Rounding")
# train the model. QDA doesn't need parameter tuning as stated in caret manual.
train_qda <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                   method = "qda",
                   data = train_samples_10k)

# test the trained model on the test data.
pred_class_qda <- predict(train_qda, test_samples_10k)
acc_qda <- confusionMatrix(pred_class_qda,
                           test_samples_10k$Classvalue)$overall["Accuracy"]

kappa_qda <- confusionMatrix(pred_class_qda,
                             test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- tibble(model="QDA", Overall_Accuracy = acc_qda, Kappa_Index = kappa_qda)

predicted_classes <- tibble(QDA = pred_class_qda)

### Linear Discriminant Analysis (LDA)
#-------------------------------------

set.seed(11, sample.kind="Rounding")
# train the model. LDA doesn't need parameter tuning as stated in caret manual.
train_lda <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                   method = "lda",
                   data = train_samples_10k)

# Test the trained model on the test data.
pred_class_lda <- predict(train_lda, test_samples_10k)
acc_lda <- confusionMatrix(pred_class_lda,
                           test_samples_10k$Classvalue)$overall["Accuracy"]

kappa_lda <- confusionMatrix(pred_class_lda,
                             test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="LDA", Overall_Accuracy = acc_lda, Kappa_Index=kappa_lda))

predicted_classes <- bind_cols(predicted_classes,tibble(LDA = pred_class_lda)) 

### Naive Bayes (NB)
#--------------------

set.seed(12, sample.kind="Rounding")
# train the model.
train_nb <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "naive_bayes",
                  data = train_samples_10k)

# test the trained model on the test data.
pred_class_nb <- predict(train_nb, test_samples_10k)
acc_nb <- confusionMatrix(pred_class_nb,
                          test_samples_10k$Classvalue)$overall["Accuracy"]
kappa_nb <- confusionMatrix(pred_class_nb,
                            test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="NB", Overall_Accuracy = acc_nb, Kappa_Index=kappa_nb))

predicted_classes <- bind_cols(predicted_classes,tibble(NB = pred_class_nb)) 

### Logistic Regression
#----------------------

set.seed(13, sample.kind="Rounding")
# train the model.
train_log <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                   method = "multinom",
                   data = train_samples_10k)

# test the trained model on the test data.
pred_class_log <- predict(train_log, test_samples_10k)
acc_log <- confusionMatrix(pred_class_log,
                           test_samples_10k$Classvalue)$overall["Accuracy"]

kappa_log <- confusionMatrix(pred_class_log,
                             test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="Logistic", Overall_Accuracy = acc_log, Kappa_Index=kappa_log))

predicted_classes <- bind_cols(predicted_classes,tibble(Logistic = pred_class_log)) 


### K-Nearest Neighbors (KNN)
#----------------------------

set.seed(14, sample.kind="Rounding")
# Set tuning parameter space to be searched
tune_grid <- data.frame(k = seq(1, 21, 2))

# train the model.
train_knn <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "knn",
                  data = train_samples_10k,
                  tuneGrid = tune_grid)

# test the trained model on the test data.
pred_class_knn <- predict(train_knn, test_samples_10k)
acc_knn <- confusionMatrix(pred_class_knn,
                test_samples_10k$Classvalue)$overall["Accuracy"]

kappa_knn <- confusionMatrix(pred_class_knn,
                test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="KNN", Overall_Accuracy = acc_knn, Kappa_Index = kappa_knn))

predicted_classes <- bind_cols(predicted_classes,tibble(KNN = pred_class_knn)) 


### Classification Tree
#--------------------------

set.seed(15, sample.kind="Rounding")

# Set complexity parameter (cP) space to be searched
tune_grid <- data.frame(cp = seq(0, 0.1, len=30))

# train the model.
train_Ctree <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "rpart",
                  data = train_samples_10k,
                  tuneGrid = tune_grid)

# test the trained model on the test data.
pred_class_Ctree <- predict(train_Ctree, test_samples_10k)
acc_Ctree <- confusionMatrix(pred_class_Ctree,
                test_samples_10k$Classvalue)$overall["Accuracy"]
kappa_Ctree <- confusionMatrix(pred_class_Ctree,
                test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="C.Tree", Overall_Accuracy = acc_Ctree, Kappa_Index=kappa_Ctree))

predicted_classes <- bind_cols(predicted_classes,tibble(C.Tree = pred_class_Ctree)) 

### Random Forest
#------------------

set.seed(16, sample.kind="Rounding")

# Set mtry parameter space to be searched
tune_grid <- data.frame(mtry = c(1, 2, 3, 4))

# train the model.
train_rf <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "rf",
                  data = train_samples_10k,
                  tuneGrid = tune_grid)

# test the trained model on the test data.
pred_class_rf <- predict(train_rf, test_samples_10k)
acc_rf <- confusionMatrix(pred_class_rf,
                test_samples_10k$Classvalue)$overall["Accuracy"]
kappa_rf <- confusionMatrix(pred_class_rf,
                test_samples_10k$Classvalue)$overall["Kappa"]


# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="RF", Overall_Accuracy = acc_rf, Kappa_Index=kappa_rf))

predicted_classes <- bind_cols(predicted_classes,tibble(RF = pred_class_rf)) 

### Support Vector Machine with Linear Kernel (SVM Linear)
#----------------------------------------------------------

set.seed(17, sample.kind="Rounding")

# Set C (cost) parameter space to be searched
tune_grid <- data.frame(C = seq(0, 30, 1))

# train the model.
train_svml <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "svmLinear",
                  data = train_samples_10k,
                  tuneGrid = tune_grid
                  )

# test the trained model on the test data.
pred_class_svml <- predict(train_svml, test_samples_10k)
acc_svml <- confusionMatrix(pred_class_svml,
                test_samples_10k$Classvalue)$overall["Accuracy"]

kappa_svml <- confusionMatrix(pred_class_svml,
                test_samples_10k$Classvalue)$overall["Kappa"]


# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="SVM_Linear", Overall_Accuracy = acc_svml, Kappa_Index = kappa_svml))

predicted_classes <- bind_cols(predicted_classes,tibble(SVM_Linear = pred_class_svml)) 

### Support Vector Machine with Radial Basis Kernel
#--------------------------------------------------

set.seed(18, sample.kind="Rounding")

# Set C (cost) parameter space to be searched
tune_grid <- data.frame(C = seq(0, 30, 3))

# train the model.
train_svmr <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "svmRadialCost",
                  data = train_samples_10k,
                  tuneGrid = tune_grid,
                  )

# test the trained model on the test data.
pred_class_svmr <- predict(train_svmr, test_samples_10k)
acc_svmr <- confusionMatrix(pred_class_svmr,
                test_samples_10k$Classvalue)$overall["Accuracy"]

kappa_svmr <- confusionMatrix(pred_class_svmr,
                test_samples_10k$Classvalue)$overall["Kappa"]


# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="SVM_Radial", Overall_Accuracy = acc_svmr, Kappa_Index = kappa_svmr))

predicted_classes <- bind_cols(predicted_classes,tibble(SVM_Radial = pred_class_svmr))

### The Ensemble Model
#-----------------------

# Select Models with accuracy larger than 0.87 (87%)
accuracy_results_88per <- accuracy_results %>% filter(Overall_Accuracy >= 0.87)
predicted_classes_88per <- predicted_classes %>%  dplyr::select(pull(accuracy_results_88per["model"]))

# Create the ensemble model prediction through majority vote by Selecting the most frequent predicted value among the models.
pred_class_ensemble <- apply(predicted_classes_88per,1, mfv1)

# Calculate the Accuracy
acc_ensemble <- confusionMatrix(as.factor(pred_class_ensemble),
                test_samples_10k$Classvalue)$overall["Accuracy"]
kappa_ensemble <- confusionMatrix(as.factor(pred_class_ensemble),
                test_samples_10k$Classvalue)$overall["Kappa"]

# Create tibbles to store the results
accuracy_results <- bind_rows(accuracy_results, tibble(model="Ensemble", Overall_Accuracy = acc_ensemble, Kappa_Index = kappa_ensemble))

predicted_classes <- bind_cols(predicted_classes,tibble(Ensemble = pred_class_ensemble))

#================================================================
# Results
#================================================================

knitr::kable(accuracy_results, caption = "Table of performance of all models")

accuracy_results %>% ggplot(aes(x=model, y=Overall_Accuracy, group=1)) + geom_line() + geom_point()


# Random Forest Model with the whole dataset

set.seed(160, sample.kind="Rounding")

# train the model.
train_rf_63k <- train(Classvalue ~ B2blue + B3green + B4red + B8nearInraRed,
                  method = "rf",
                  data = train_samples_63k)

# test the trained model on the test data.
pred_class_rf_63k <- predict(train_rf_63k, test_samples_63k)
conf_mat_rf_63k <- confusionMatrix(pred_class_rf_63k,
                test_samples_63k$Classvalue)

print(conf_mat_rf_63k)

## Classify the Whole Image

classified <- predict(cropped,train_rf_63k, type = "raw")
plot(classified)
writeRaster(classified,paste0("./outputs/classifiedImage_RF",".tif"),
            format="GTiff", overwrite = TRUE)
