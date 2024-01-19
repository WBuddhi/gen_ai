# Databricks notebook source
!pip install bert-score rouge_score

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

df = spark.table("llama2_feature_store.pls.test_predictions_gpt_summary_max_token_1000")

# COMMAND ----------

rows = df.select(["prediction", "summary", "article","gpt_summary"]).collect()
predictions = [row.prediction for row in rows]
reference = [row.summary for row in rows]

# COMMAND ----------

article = [row.article for row in rows]
gpt_summary = [row.gpt_summary for row in rows]

# COMMAND ----------

from evaluate import load

# COMMAND ----------

bertscore = load("bertscore")
results = bertscore.compute(predictions=predictions, references=reference, batch_size = 10, lang="en")

# COMMAND ----------

results

# COMMAND ----------

rouge = load('rouge')
results = rouge.compute(predictions=predictions, references=reference)

# COMMAND ----------

results

# COMMAND ----------

print(article[15])

# COMMAND ----------

print(gpt_summary[15])

# COMMAND ----------

print(predictions[15])

# COMMAND ----------

reference[15]

# COMMAND ----------


