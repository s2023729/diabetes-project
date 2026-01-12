# ===============================
# 0. LIBRARIES
# ===============================
library(tidyverse)
library(caret)
library(pROC)
library(naivebayes)
library(rpart)

# ===============================
# 1. LOAD DATA
# ===============================
df_clean <- read_csv("diabetes_dataset.csv")

# ===============================
# 2. DATA PREPARATION
# ===============================
df_model <- df_clean %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(diagnosed_diabetes = as.factor(diagnosed_diabetes)) %>%
  select(-diabetes_stage, -diabetes_risk_score)

# caret needs valid class labels
df_model$diagnosed_diabetes <- factor(
  df_model$diagnosed_diabetes,
  levels = c(0, 1),
  labels = c("No", "Yes")
)

# ===============================
# 3. TRAIN–TEST SPLIT
# ===============================
set.seed(123)
index <- createDataPartition(df_model$diagnosed_diabetes, p = 0.7, list = FALSE)
train_data <- df_model[index, ]
test_data  <- df_model[-index, ]

# ===============================
# 4. CROSS-VALIDATION SETUP
# ===============================
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE
)

# ===============================
# 5. MODELS
# ===============================
set.seed(123)
model_glm <- train(
  diagnosed_diabetes ~ ., data = train_data,
  method = "glm", family = "binomial",
  trControl = ctrl
)

set.seed(123)
model_tree <- train(
  diagnosed_diabetes ~ ., data = train_data,
  method = "rpart",
  trControl = ctrl
)

set.seed(123)
model_nb <- train(
  diagnosed_diabetes ~ ., data = train_data,
  method = "naive_bayes",
  trControl = ctrl
)

set.seed(123)
model_rf <- train(
  diagnosed_diabetes ~ ., data = train_data,
  method = "rf",
  ntree = 50,
  trControl = ctrl
)

# ===============================
# 6. EVALUATION FUNCTION
# ===============================
get_all_metrics <- function(model, test_data) {
  pred_class <- predict(model, test_data, type = "raw")
  pred_prob  <- predict(model, test_data, type = "prob")
  
  cm <- confusionMatrix(
    pred_class,
    test_data$diagnosed_diabetes,
    mode = "everything"
  )
  
  roc_obj <- roc(
    test_data$diagnosed_diabetes,
    pred_prob$Yes,
    quiet = TRUE
  )
  
  c(
    Accuracy    = cm$overall["Accuracy"],
    Kappa       = cm$overall["Kappa"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision   = cm$byClass["Precision"],
    F1_Score    = cm$byClass["F1"],
    AUC         = auc(roc_obj)
  )
}

# ===============================
# 7. FINAL COMPARISON
# ===============================
results_glm  <- get_all_metrics(model_glm, test_data)
results_tree <- get_all_metrics(model_tree, test_data)
results_nb   <- get_all_metrics(model_nb, test_data)
results_rf   <- get_all_metrics(model_rf, test_data)

final_comparison <- data.frame(rbind(
  LogisticReg  = results_glm,
  DecisionTree = results_tree,
  NaiveBayes   = results_nb,
  RandomForest = results_rf
))

print(round(final_comparison, 3))