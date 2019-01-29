library(tidyverse)
library(readxl)
library(lattice)
#library(BDbasics)
library(caret)
library(Hmisc)
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
library(snow)
library(doSNOW)
library(e1071)
library(ranger)
library(Boruta)
library(GGally)
library(Matrix)
library(glmnet)


# load serialized data ----------------------------------------------------
# load("Rdata/data-SQ.RData")
sqAnoDf <- read_tsv(file = "anonymized-sq-dataset.tsv")

str(sqAnoDf)
dim(sqAnoDf)
table(sqAnoDf[ ,"LABEL"])
# load training configuration ---------------------------------------------
trainConfigDf <- read_xlsx(path = "Training Configuration.xlsx",
                           sheet = "Sheet1") %>%
  dplyr::filter(use == 1)


# set-up ------------------------------------------------------------------
useSeed <- TRUE
nSeed <- 123456789
nCores <- detectCores()
pSubSample <- 0.1
indResp <- 1

sumList <- list()
plotList <- list()
fitList <- list()


# list predictive features to be included in training process -------------
# features are organized by rows of 5
responsesOfInterest <- c("LABEL")
featuresOfInterest <-  paste0("X", 1:144)


# load data set and give generic name -------------------------------------
if(useSeed)
  set.seed(nSeed)

compDf <- sqAnoDf %>%
  dplyr::mutate(LABEL = as.factor(LABEL))%>%
  dplyr::select(responsesOfInterest,
                featuresOfInterest) %>%
  na.omit(.) %>%
  dplyr::group_by_(responsesOfInterest[indResp]) %>%
  dplyr::sample_frac(pSubSample) %>%
  dplyr::ungroup()

classResp <- class(compDf[, responsesOfInterest[indResp]][[1]])


# Separate training data and test data ------------------------------------
set.seed(123)
partition_indexes <- createDataPartition(compDf$LABEL, times = 1, p = 0.75, list = FALSE)
compDf.train <- compDf[partition_indexes, ]
compDf.test <- compDf[-partition_indexes, ]


# sanity check ------------------------------------------------------------
table(compDf[, responsesOfInterest[indResp]][[1]])
#  Low High 
# 3611  963 


# summary plot ------------------------------------------------------------
# plot(compDf)
plotList[["corPlot"]] <- ggpairs(compDf,
                                 upper = list(continuous = wrap('cor',
                                                                method = "spearman")),
                                 mapping = ggplot2::aes_string(colour = responsesOfInterest[indResp])) +
  theme_bw()

plotList[["corPlot"]]


# summary stats -----------------------------------------------------------


# per level of response if either factor or character


# Boruta ------------------------------------------------------------------
if(useSeed)
  set.seed(nSeed)

ptm <- Sys.time()
resBoruta <- Boruta(x = compDf.train[, featuresOfInterest],
                    y = compDf.train[, responsesOfInterest[indResp]][[1]],
                    doTrace = 2,
                    num.threads = nCores - 1,
                    ntree = 200)
timer(ptm)
print(resBoruta)

selected<-getSelectedAttributes(resBoruta)

resBorutaImpDf <- as.data.frame(resBoruta$ImpHistory) %>%
  tidyr::gather(key = "Feature", value = "Importance") %>%
  dplyr::mutate(Decision = as.character(resBoruta$finalDecision[match(Feature,
                                                                      names(resBoruta$finalDecision))]),
                Decision = factor(ifelse(grepl("shadow", Feature),
                                         "Shadow",
                                         Decision),
                                  levels = c("Confirmed", "Tentative", "Rejected", "Shadow")))


sumImpSqBorutaDf <- resBorutaImpDf %>%
  dplyr::group_by(Feature) %>%
  dplyr::summarize(Importance = median(Importance,
                                       na.rm = TRUE),
                   Decision = unique(Decision)) %>%
  dplyr::arrange(desc(Importance)) %>%
  dplyr::mutate(Rank = 1:n())

xLevels <- as.character((resBorutaImpDf %>%
                           dplyr::group_by(Feature) %>%
                           dplyr::summarize(median = median(Importance,
                                                            na.rm = TRUE)) %>%
                           dplyr::arrange(median) %>%
                           as.data.frame(.))$Feature)

plotList[["Boruta"]] <- ggplot(data = resBorutaImpDf %>%
                                 dplyr::mutate(Feature = factor(Feature,
                                                                levels = xLevels)),
                               aes(x = Feature,
                                   y = Importance,
                                   fill = Decision)) +
  geom_boxplot() +
  scale_fill_manual(values = c("green", "yellow", "red", "blue"),
                    drop = FALSE) +
  coord_flip() +
  theme(axis.text.x = element_text(size = 8,
                                   hjust = 1,
                                   vjust = 1)) +
  ggtitle("Output of Boruta Algorithm") +
  theme_bw()

plotList[["Boruta"]]


# caret processes ---------------------------------------------------------
# parallelized with foreach and doParallel package
cl <- makePSOCKcluster(nCores - 1)
registerDoParallel(cl)

fitList <- foreach(iCond = 1:nrow(trainConfigDf)) %do% {
  feMethod <- trainConfigDf$method[iCond]
  fePreProc <- if(!is.na(trainConfigDf$preProc[iCond])){
    unlist(strsplit(trainConfigDf$preProc[iCond], split = "\\s"))
  }else{
    NULL
  }
  
  feTrainMetric <- trainConfigDf$trainMetric[iCond]
  
  feTrainControl <-  if(!is.na(trainConfigDf$trainControl[iCond])){
    eval(parse(text = paste0("trainControl(",
                             trainConfigDf$trainControl[iCond],
                             ")")))
  }else{
    trainControl()
  }
  
  
  feTuneGrid <- if(!is.na(trainConfigDf$tuneGrid[iCond])){
    eval(parse(text = paste0("expand.grid(",
                             trainConfigDf$tuneGrid[iCond],
                             ")")))
  }else{
    NULL
  }
  
  if(useSeed)
    set.seed(nSeed)
  
  train(as.formula(paste0(responsesOfInterest[indResp],
                          " ~",
                          paste(featuresOfInterest,
                                collapse = "+"))),
        data = compDf.train,
        method = feMethod,
        preProcess = fePreProc,
        tuneGrid =  feTuneGrid,
        metric = feTrainMetric,
        trControl = feTrainControl)
  

  
}
names(fitList) <- trainConfigDf$name
stopCluster(cl)


# summarize performance ---------------------------------------------------
reSampledFit <- resamples(fitList)
summary(reSampledFit)

plotList[["dotPlot"]] <- dotplot(reSampledFit,
                                 metric = feTrainMetric) #temporary: 
#uses latest choice in foreach loop

plotList[["dotPlot"]]

gc()


# use best model ----------------------------------------------------------
method<-trainConfigDf$name

i=1
while(i<=length(method)) {
  preds<-predict(fitList[[method[i]]],compDf.test)
  return(confusionMatrix(preds, compDf.test$LABEL))
  i<-i+1
}

preds<-predict(fitList[["XGBoost"]],compDf.test)
confusionMatrix(preds, compDf.test$LABEL)



# serialize results -------------------------------------------------------
save(sumList,
     plotList,
     fitList,
     reSampledFit,
     file = "analysisCaret.RData")
load(file = "analysisCaret.RData")


# all methods -------------------------------------------------------------
# knn
timer <- function(start_time) {
  start_time <- as.POSIXct(start_time)
  dt <- difftime(Sys.time(), start_time, units="secs")
  format(.POSIXct(dt,tz="GMT"), "%H:%M:%S")
}

train.control <- trainControl(method = "cv", number = 10, search = "grid")
tune.grid.knn <- expand.grid(k = c(2,3,4,5,6,7,8,9,10))
ptm <- Sys.time()
caret.knn <- train(LABEL ~ ., data = compDf.train,preProcess=c("center","scale"),tuneGrid=tune.grid.knn,method = "knn", trControl = train.control)
timer(ptm)

caret.knn
preds.knn <- predict(caret.knn, compDf.test)
confusionMatrix(preds.knn, compDf.test$LABEL)

#Ranger
if(useSeed)
  set.seed(nSeed)
xanger.params <- getModelInfo("ranger")
xanger.params$ranger$parameters
ptm <- Sys.time()
caret.ranger <- train(LABEL ~ .,data = compDf.train,method = "ranger", trControl = train.control)
timer(ptm)

caret.ranger
preds.ranger <- predict(caret.ranger, compDf.test)
confusionMatrix(preds.ranger, compDf.test$LABEL)

#rpart
xanger.params <- getModelInfo("rpart")
xanger.params$rpart$parameters
ptm <- Sys.time()
caret.rpart <- train(LABEL ~ .,data = compDf.train,method = "rpart", trControl = train.control)
timer(ptm)

caret.rpart
preds.rpart <- predict(caret.rpart, compDf.test)
confusionMatrix(preds.rpart, compDf.test$LABEL)

#XgbTree
xgb.params <- getModelInfo("xgbTree")
xgb.params$xgbTree$parameters
tune.grid.xgb <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         max_depth = c(6,8,10),
                         min_child_weight = c(2.0, 2.25, 2.5),
                         gamma = c(0.001, 0.01, 0.1))
ptm <- Sys.time()
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
caret.xgb <- train(LABEL ~ ., data = compDf.train, tuneGrid=tune.grid.xgb,method = "xgbTree", trControl = train.control)
stopCluster(cl)
timer(ptm)

caret.xgb
preds <- predict(caret.xgb, compDf.test)
confusionMatrix(preds, compDf.test$LABEL)

#glmnet
xgb.params <- getModelInfo("glmnet")
xgb.params$glmnet$parameters
tune.grid.glmnet <- expand.grid(alpha =c(0,1),
                             lambda=seq(0,0.5,0.001))
ptm <- Sys.time()
caret.glmnet <- train(LABEL ~ ., data = compDf.train,tuneGrid=tune.grid.glmnet,method = "glmnet", trControl = train.control)
timer(ptm)

caret.glmnet
preds <- predict(caret.glmnet, compDf.test)
confusionMatrix(preds, compDf.test$LABEL)


# newdata -----------------------------------------------------------------
newdata.train<-compDf.train[selected]
newdata.test<-compDf.test[selected]
