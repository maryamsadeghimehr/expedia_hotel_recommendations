## Import the libraries
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import pyspark.sql
import numpy as np
from pyspark.sql import functions as F
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes,GBTClassifier
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import types

from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from sklearn.decomposition import truncated_svd
###############################################
spark = SparkSession.builder.appName('Hotel_Recommendations').getOrCreate()
###############################################
from pysparkling import *
import h2o
h2o.init()
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
##############################################
hc = H2OContext.getOrCreate()


############################################
def get_data():
    
    train = spark.read.csv('../data/train.csv', header = True,multiLine=True)
    test = spark.read.csv('../data/test.csv', header = True,multiLine= True)
    
    train = train.coalesce(100)
    test = test.coalesce(100)
    train = train.where(col('is_booking') == 1).sample(0.015, seed=42)
#     train = train.where((col('is_booking') == 1)&
#                        (col('orig_destination_distance').isNotNull())).sample(0.015, seed=42)
    test = test.sample(0.001, seed=42)

    return train, test


##############################################
def combine_train_test(train, test,col_diff_list,test_col_drop):
    for col in col_diff_list:
        test = test.withColumn(col, F.lit(None).cast(StringType()))
    df = train.unionByName(test.drop(test_col_drop))
    
    return df


#############################################
def print_missing(df):
    
    for i in df.columns:
        print(i, df.where(col(i).isNull()).count())
    
    return 0
    

#############################################
def split_TRAIN_TEST(df):
    
    df = df.withColumn('rnd', F.rand())
#     df = (df.withColumn('split', 
#                         F.expr('IF("rnd" < 0.8, "TRAIN", "TEST")')
#                        )
#           #.drop('rnd')
#          )
    df=df.withColumn('split', F.when(F.col('rnd')< 0.8,"TRAIN")
                                    .otherwise("TEST")
                    ).drop('rnd')
    
    return df


##########################################
def split_TRAIN_TEST_combined(df):
    
    df=df.withColumn('split', F.when((F.col('hotel_cluster').isNotNull())
                                     ,"TRAIN")
                                    .otherwise("TEST")
                    )
    
    return df

##########################################
def semi_target_encoding(df, col, target):
    
    tmp1 = (df
        .groupBy(col)
        .pivot(target)
        .agg(F.count(target).alias('cnt'))
        .fillna(0)
        .join(df
              .groupBy(col)
              .agg(F.count('*').alias('total')),
              on=col,
              how ='left')
       )
    for i in range(100):
        tmp1 = tmp1.withColumn(str(i), F.col(str(i)) / F.col('total'))

    df = df.join(tmp1, on = col, how = 'left')
    
    return df


##########################################
def semi_target_encoding1(df, col, target):
    
    tmp1 = df.groupBy(target, col).count()
    tmp2 = df.groupBy(col).agg(F.count(col).alias(col+'_Cnt'))
    tmp3 = tmp1.join(tmp2,on =col).withColumn('rate', F.col('count')/F.col(col+'_Cnt'))
    
    for i in range(100):
        df = df.withColumn(str(i),F.lit(0))
    
    df = df.join(tmp3.drop(target, 'count', col+'_Cnt'), on = col, how = 'left').drop(col)
    for i in range(100):
        df = df.withColumn(
            str(i), F.expr('IF(hotel_cluster = {}, rate, 0)'.format(i))
        ).drop(col,'rate')
    df.cache().count()
    
    return df


##########################################
def type_to_integer(df, split_col):
    for col in df.drop(split_col).columns:
        df = df.withColumn(col, df[col].astype('int'))
    
    return df


########################################
def build_models(i,labelCol,features_list, train, test):
    
    model_list = [
      LogisticRegression(featuresCol='features', labelCol = labelCol),
      RandomForestClassifier(featuresCol='features', labelCol = labelCol),
      NaiveBayes(featuresCol='features',labelCol=labelCol), 
      GBTClassifier(featuresCol='features',labelCol=labelCol)
    ]
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    pipeline = Pipeline(stages=[assembler, model_list[i]])
    model = pipeline.fit(train)
    prediction = model.transform(test)
    
    return prediction


########################################
def pred_precision(prediction):
    
    pred_label = prediction.rdd.map(lambda 
                                    x: (float(x.prediction),
                                        float(x.hotel_cluster)))
    metrics = MulticlassMetrics(pred_label)
    precision = metrics.precision()
    
    return round(precision*100,2)


######################################
def pred_precision_kaggle(prediction,NumCluster):
    
    pred_label = prediction.rdd.map(lambda
                                    x: (float(np.argsort(-1*x.probability)[:1]),
                                        float((x.hotel_cluster))))
    metrics = MulticlassMetrics(pred_label)
    avg_precision = metrics.precision()
    
    for i in range(1,NumCluster):
        pred_label = prediction.rdd.map(lambda
                                        x: (float(np.argsort(-1*x.probability)[i:(i+1)]),
                                            float(x.hotel_cluster)))
        metrics = MulticlassMetrics(pred_label)
        avg_precision += metrics.precision()
        
    return avg_precision


##########################################
def eval_kaggle_score(df_pred, Num):
    metrics = MulticlassMetrics(df_pred.rdd.map(lambda ar: (float(np.argsort(ar.probability)[-1:]), float(ar.hotel_cluster))))
    NumCluster = Num
    avg_precision = metrics.precision()
    for i in range(1, NumCluster):
        metrics = MulticlassMetrics(df_pred.rdd.map(lambda ar: (float(np.argsort(ar.probability)[-(i+1):-i]), float(ar.hotel_cluster))))
        avg_precision += metrics.precision()
    return avg_precision


############################################
### Popularity
def hotel_popularity(df,col1, col2):
    
    hotel_pivot = df.groupby(col1).pivot(col2).count().na.fill(0)
    hotel_assembler = VectorAssembler(inputCols=[str(i) for i in range(100)],
                                     outputCol='popular_hotels')
    hotel_popular = hotel_assembler.transform(hotel_pivot)
    
    @F.udf(returnType = types.StringType())
    def findTopN(i,x):
        
        return (np.argsort(x)[-(i+1):-i]).tolist()
    
    for i in range(1,6):
        hotel_popular = hotel_popular.withColumn('popular_hotels'+str(i), 
                                                  F.regexp_extract(
                                                      findTopN(F.lit(i),hotel_popular.popular_hotels),
                                                    '[0-9]+',
                                                      0
                                                  )
                                                  .astype('int'))
    hotel_popular = hotel_popular.drop(*[str(i) for i in range(100)],'popular_hotels')
    
    df = df.join(hotel_popular, on = 'srch_destination_id', how = 'left')
    df.cache().count()
    
    return df


##########################################
def build_h2o_models(i,labelCol, features_list, df_train, df_test):
                     
                     h2o_train = hc.asH2OFrame(df_train)
                     h2o_test = hc.asH2OFrame(df_test)
                     
                     models_list = [H2ORandomForestEstimator(ntrees=20, max_depth=10, nfolds=10),
                                   H2OGradientBoostingEstimator(ntrees = 50, max_depth=20)]
                     
                     model = models_list[i]
                     model.train(x=features_list, y=labelCol, training_frame=h2o_train);
                     performance = model.model_performance(test_data=h2o_test);
                     pred = model.predict(h2o_test)
                     
                     return performance,pred,model,h2o_test
                

                
######################################
def encoding(i,df,col):
    
    encoder = OneHotEncoderEstimator(
        inputCols= [col],
        outputCols=["p"+str(i)]
    )
    encoder = encoder.fit(df)
    df = encoder.transform(df)
    
    return df#, encoder
  
    
####################################
def encoding2(df,incol,outcol):
    
    encoder = OneHotEncoderEstimator(
        inputCols= [incol],
        outputCols=[outcol]
    )
    encoder = encoder.fit(df)
    df = encoder.transform(df)
    
    return df#, encoder


#####################################
def target_encoding(df, col, target):
    
    avg = df.groupBy(col).agg(F.count(col).alias(col + 'Cnt'))
    avg= avg.withColumn(
        col + 'PerCnt',
        F.col(col + 'Cnt')/
        int(avg.agg(F.sum(F.col(col + 'Cnt'))
                   ).collect()[0][0]))
    df = (df
          .join(avg, on = col, how = 'left')
          .drop(col,col + 'Cnt')
         )
                 
    return df


###############################
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(20, 18))
    
    sns.heatmap(
        df.corr(), 
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    

###################################
def make_positive(df, col):
    makePos = F.udf(lambda x: x if x > 0 else 1)
    df = df.withColumn(col, makePos(df[col]))
    
    return df


##################################
def parse_date(df):
    
    df = df.withColumn('srch_ci_day',F.dayofweek(df['srch_ci']))
    df = df.withColumn('srch_ci_dayofmonth',F.dayofmonth(df['srch_ci']))
    df = df.withColumn('srch_ci_month',F.month(df['srch_ci']))
    df = df.withColumn('srch_ci_year',F.year(df['srch_ci']))

    df = df.withColumn('srch_co_day',F.dayofweek(df['srch_co']) )
    df = df.withColumn('srch_co_dayofmonth',F.dayofmonth(df['srch_co']) )
    df = df.withColumn('srch_co_month',F.month(df['srch_co']) )
    df = df.withColumn('is_weekend',
             (
                 (df['srch_ci_day'] == 6)| (df['srch_ci_day'] == 7) |
                 (df['srch_co_day'] == 6)| (df['srch_co_day'] == 7) 
             ).astype('int')
             )
    return df


################################
def build_cv_models(i,labelCol,features_list, train, test,n_fold):
    
    model_list = [
        LogisticRegression(featuresCol='features', labelCol = labelCol),
        RandomForestClassifier(featuresCol='features', labelCol = labelCol),
        NaiveBayes(featuresCol='features',labelCol=labelCol), 
        ]

    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    pipeline = Pipeline(stages=[assembler, model_list[i]])
    
    paramGrid = [(ParamGridBuilder()
                 .addGrid(model_list[0].regParam, [0.0])
                 .addGrid(model_list[0].fitIntercept, [True])
                 .addGrid(model_list[0].elasticNetParam, [0.0])
                  .build()
                 ),
                 (ParamGridBuilder()
                 .addGrid(model_list[1].maxDepth, [5])
                 .addGrid(model_list[1].numTrees, [20])
                  .build()
                )
                ]
    evaluator = [
        RegressionEvaluator(labelCol=labelCol,predictionCol='prediction'),
        MulticlassClassificationEvaluator(labelCol=labelCol,predictionCol='prediction')
    ]
    
                 
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid[i],
                         evaluator= evaluator[i],
                         numFolds=n_fold)
    model = crossval.fit(train)
    pred = model.transform(test)
    
    return model, pred


##########################################
def combine_train_test(train, test,col_diff_list):
    for col in col_diff_list:
        test = test.withColumn(col, F.lit(None).cast(StringType()))
    df = train.unionByName(test.drop('id'))
    
    return df


#######################################
    