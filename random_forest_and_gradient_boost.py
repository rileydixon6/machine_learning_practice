# Databricks notebook source
# MAGIC %md
# MAGIC # Machine Learning: Anonymous Penquins

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import PipelineModel
import pandas as pd
from plotnine import *
import pyspark.sql.functions as F
!pip install altair
import altair as alt

# COMMAND ----------

# https://kb.databricks.com/machine-learning/extract-feature-info.html
def importance_plot(pipelineModel):
  va = pipelineModel.stages[-3] #### needs to be the "vector assembler" stage in your pipeline" ####
  tree = pipelineModel.stages[-1].bestModel ##### is the cv stage of your pipeline ####
  a = list(zip(va.getInputCols(), tree.featureImportances))
  dat_fi = pd.DataFrame({'variable': [i[0] for i in a], 'importance':[i[1] for i in a]}) \
    .sort_values('importance', ascending=False) \
    .assign(
      variable_cat = lambda x: pd.Categorical(x['variable'], categories = x.sort_values('importance', ascending=True)['variable']))

  plot = ggplot(dat_fi, aes(y = "importance", x = "variable_cat")) +\
    geom_col(fill = "lightblue") +\
    coord_flip() +\
    theme_bw() +\
    labs(title = "Feature importance plot", x = "Feature")
  print(plot)
  return dat_fi

def model_results(predDF, labelCol = 'diff', predictCol = 'prediction'):
  dat = predDF.select(labelCol, predictCol, 'number_months_available').toPandas()
  
  evaluator = RegressionEvaluator(labelCol="diff", predictionCol="prediction", metricName="rmse")
  rmse = evaluator.evaluate(predDF)
  rmse = "RMSE data = %g" % rmse
  rsquared = "  R-squared:" + str(round(predDF.stat.corr('diff', 'prediction'),2))

  
  p = ggplot(dat, aes(x = labelCol, y = predictCol, color = 'factor(number_months_available)')) +\
    geom_point(alpha = .7, size = 2) +\
    geom_abline(intercept=0, slope=1, color = "darkgrey") +\
    theme_bw() +\
    scale_color_brewer(type="qual", palette="Dark2") +\
    labs(title = "Predicted Vs. Actual:\n" +  rmse + rsquared, 
         color = "Number of Months Available", 
         x = "Actual Values \n (Difference in Checkouts from Current Month to Next Month)", 
         y = "Predicted Values")
  
  return p

model = PipelineModel.load('dbfs:/gbt_trial_2021_06_07_03_05')
fmModel = PipelineModel.load("dbfs:/fmModel")
df = spark.sql('SELECT * FROM library_features.ml_penguins_final')
train_final = spark.sql('SELECT * FROM library_features.ml_penguins_train')
test_final = spark.sql('SELECT * FROM library_features.ml_penguins_test')
pred_final_dataset = model.transform(test_final)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Factorization Machine Regressor
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC  Factorization Machines are able to estimate interactions between features even in problems with huge sparsity (like advertising and recommendation system).
# MAGIC 
# MAGIC  FM can be used for regression
# MAGIC 
# MAGIC  FM also can be used for binary classification through sigmoid function. The optimization criterion is logistic loss.
# MAGIC 
# MAGIC  RMSE = 75

# COMMAND ----------

# from sklearn.preprocessing import MinMaxScaler

# def importance_plot1(pipelineModel):
#   va = pipelineModel.stages[0]
#   tree = pipelineModel.stages[-1]
#   a = list(zip(va.getInputCols(), tree.featureImportances))
#   dat_fi = pd.DataFrame({'variable': [i[0] for i in a], 'importance':[i[1] for i in a]}) \
#     .sort_values('importance', ascending=False) \
#     .assign(
#       variable_cat = lambda x: pd.Categorical(x['variable'], categories = x.sort_values('importance', ascending=True)['variable']))

# print(model_results(fmModel.stages[-1]))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # THE MODEL
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Gradient Boosted Tree
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC - Gradient Boosted Tree's utilize a series of weak learners, in this case very shallow decision trees, to produce a strong learner.
# MAGIC - They require very little preprocessing of data. They handle categorical and numerical data, and aren't affected by scaling.
# MAGIC - They can be very finely tuned through hyperparameters for the specific data they are training on, using a grid search with a gradient boosted tree helps to find the right parameters for your data.
# MAGIC 
# MAGIC - They can be weak to overfitting, k-fold cross-validation alleviates this issue.
# MAGIC - The large amount of grid parameters needed to really tune a gradient boosted tree **can** take a lot of computational time training many combinations of models to find the right one for your data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # The Target
# MAGIC 
# MAGIC ### Difference between 'target' and 'current_month_checkouts'
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC  * Target is the difference of next month's checkouts from this month's checkouts to prevent auto-correlation.
# MAGIC  * This target will help the library to know how quickly a book's checkouts is decreasing so they can sell their books off sooner.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Feature Importance
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC This model uses all the featues in the `model_months_all` table, plus 3 additional features our class engineered.
# MAGIC 
# MAGIC Only three additional engineered features made any significant impact on the model:
# MAGIC 
# MAGIC * seasonal_strength: The max avg zscore over two years of a book. Higher values indicate that a book is seasonal.
# MAGIC * for_teens: This book is intended for teens
# MAGIC * for_kids: This book is intended for kids
# MAGIC   * Note: There can be overlap between those two features.
# MAGIC 
# MAGIC Finding more features that have an influence on our model would be beneficial to overall accuracy.

# COMMAND ----------

importance_plot(model)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # MODEL PERFORMANCE
# MAGIC 
# MAGIC We used a small grid search combined with k-fold cross-validation to find good parameters for our dataset, unfortunatly grid searches take a very long time, and this model can likely be further optimized than we have been able to.

# COMMAND ----------

print(model_results(pred_final_dataset ))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # PATTERNS OF WELL FITTING AND POOR FITTING
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC * **6 month** data is very accurate
# MAGIC * **3 month** data is much more difficult to predict
# MAGIC * **2 month** data produces the most extreme incorrect predictions, and is rarely accurately predicted

# COMMAND ----------

import altair as alt
dat = pred_final_dataset.withColumn('distance', F.abs(F.col('diff') - F.col('prediction'))).drop('squishedFeatures', 'features').toPandas()

# COMMAND ----------

# DBTITLE 1,Do seasonal books have less accurate predictions?
chart = alt.Chart(dat).encode(
  y = alt.Y(
    'distance',
    title='Predictive Inaccuracy'
  ),
  x = alt.X(
    'seasonal_strength',
    title='Strength of Seasonality'
  ),
  color = alt.Color('number_months_available:O', 
                    scale=alt.Scale(scheme='dark2'), 
                    title = "Number of Months Available")
).properties(title='Seasonal Strength vs Accuracy').mark_circle()
chart

# COMMAND ----------

# DBTITLE 1,Does monthly inventory of a book impact prediction accuracy?
chart3 = alt.Chart(dat).encode(
  y = alt.Y(
    'distance',
    title='Predictive Inaccuracy'
  ),
  x = alt.X(
    'total_collection_other',
    title='Total Collection Other'
  ),
  color = alt.Color('number_months_available:O', 
                    scale=alt.Scale(scheme='dark2'), 
                    title = "Number of Months Available")
).properties(title='Inventory vs Accuracy').mark_circle()
chart3

# COMMAND ----------

# DBTITLE 1,Do books with higher checkouts have more accurate predictions?
chart2 = alt.Chart(dat).encode(
  y = alt.Y(
    'distance',
    title='Predictive Inaccuracy'
  ),
  x = alt.X(
    'current_month_checkouts',
    title='Number of Checkouts this month'
  ),
  color = alt.Color('number_months_available:O', 
                    scale=alt.Scale(scheme='dark2'), 
                    title = "Number of Months Available")
).properties(title='Current Checkouts vs Accuracy').mark_circle()
chart2

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ---

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Not So Good Random Forest Model
rfg = PipelineModel.load("dbfs:/Chris_model")
pred_final_dataset1 = rfg.transform(test_final)

print(model_results(pred_final_dataset1 ))
importance_plot(rfg)

# COMMAND ----------

# DBTITLE 1,Shapely graphs for Random Forest
import shap
!pip install shapely
from shapely import *
!pip install shparkley


shap_values = shap.TreeExplainer(rfg.stages[-1].bestModel).shap_values(train_final.toPandas())

shap.summary_plot(shap_values, train_final.toPandas(), plot_type="bar")



# COMMAND ----------

shap.summary_plot(shap_values, train_final.toPandas())

# COMMAND ----------

shap_interaction_values = shap.TreeExplainer(rfg.stages[-1].bestModel).shap_interaction_values(train_final.toPandas().iloc[:2000,:])

shap.summary_plot(shap_interaction_values, train_final.toPandas().iloc[:2000,:])



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # GO NO GO
# MAGIC 
# MAGIC No go, our model is worse than guessing at a books most volatile moments which is our primary interest to predict
