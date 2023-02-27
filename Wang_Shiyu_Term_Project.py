import sparknlp
# spark = sparknlp.start() 
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import pandas as pd

import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession

from sklearn.metrics import classification_report,accuracy_score

spark = sparknlp.start()
print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

#read dataset and split it into training and testing set
dataset = spark.read.option("header", False).csv(sys.argv[1]).toDF('sentiment', 'id', 'date', 'query', 'user_id', 'text')
dataset = dataset.drop('id', 'date', 'query', 'user_id')
splitedDataset = dataset.randomSplit([0.8, 0.2], seed=26)
trainSet = splitedDataset[0]
testSet = splitedDataset[1]

# change the text content into document type
document = DocumentAssembler().setInputCol("text").setOutputCol("document")

# we can also use sentence detector here 
# if we want to train on and get predictions for each sentence
# downloading pretrained embeddings
bert_emb = BertSentenceEmbeddings.pretrained('sent_small_bert_L8_512')\
  .setInputCols(['document'])\
  .setOutputCol('sentence_embeddings')

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("sentiment")\
  .setMaxEpochs(20)\
  .setEnableOutputLogs(True)

model_pipeline = Pipeline(stages = [document, bert_emb, classsifierdl])
model = model_pipeline.fit(trainSet)


pred = model.transform(testSet).select('sentiment','text','class.result').toPandas()
pred['result'] = pred['result'].apply(lambda x:x[0])
print(classification_report(pred.sentiment,pred.result))
print(accuracy_score(pred.sentiment,pred.result))

