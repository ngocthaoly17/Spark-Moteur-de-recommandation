spark.conf.set(
  "fs.azure.account.key.storagestudent.blob.core.windows.net",
  "pH3rgal+XcwJXc3hQEYEAE+dMBo6YzhKnb4iYQNlTZ9lXaxe8RWmZwVPMF1j2V5zwBnBZ/iNu8JoFgApOxdn4Q=="
)

datasets = {
  dataset: spark.read.load(
    "wasbs://default@storagestudent.blob.core.windows.net/datasets/S8-3/Exo/restaurant-data-with-consumer-ratings/{0}.csv".format(dataset),
    format="csv",
    header="true"
  )
  for dataset in [
    "chefmozaccepts",
    "chefmozcuisine",
    "chefmozhours4",
    "chefmozparking",
    "geoplaces2",
    "rating_final",
    "usercuisine",
    "userpayment",
    "userprofile"
  ]
}


"""Créez un moteur de recommandation en utilisant les notes globales des utilisateurs
pour les restaurants et générez des prédictions pour l’ensemble de test. Servez vous
du dataset rating.
2. A l’aide des datasets userpayment et chefmozaccepts, filtrez le dataframe contenant
les prédictions pour l’ensemble de test pour retirer les recommandations concernant
des users et des restaurants dont les moyens de paiement déclarés ne sont pas
compatible.
3. Effectuez une évaluation de votre modèle sur le dataframe que vous venez de filtrer."""


from pyspark.sql.types import *
from pyspark.sql import functions as F

userpayment = datasets["userpayment"]
chefmozaccepts = datasets["chefmozaccepts"]
rating = datasets["rating_final"]

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


display(rating)


rating = rating.select(
  F.col("placeID").cast(IntegerType()),
  F.col("rating").cast(IntegerType()),
  F.col("userID")
)


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="userID", outputCol="user_ID")
indexed_rating = indexer.fit(rating).transform(rating)



train, test = indexed_rating.randomSplit([0.7, 0.3])

als = ALS(userCol="user_ID", itemCol="placeID", ratingCol="rating", coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

paramGrid = ParamGridBuilder() \
                    .addGrid(als.rank, [1, 5]) \
                    .addGrid(als.maxIter, [5, 10]) \
                    .addGrid(als.regParam, [0.3, 0.1, 0.01]) \
                    .addGrid(als.alpha, [2.0,3.0]) \
                    .build()

cv = CrossValidator(
     estimator=als,
     evaluator=evaluator,
     estimatorParamMaps=paramGrid,
     numFolds=5
)

model = cv.fit(train)


predictions = model.transform(test)
evaluator.evaluate(predictions)

display(predictions)


predictions = predictions.withColumn("prediction", F.abs(F.round(predictions["prediction"],0)))
display(predictions)

userRecommendations = model.bestModel.recommendForAllUsers(10)
display(userRecommendations)

itemRecommendations = model.bestModel.recommendForAllItems(10)
display(itemRecommendations)

display(userpayment)

display(chefmozaccepts)

chefmozaccepts =  chefmozaccepts.withColumnRenamed("Rpayment", "Upayment")

display(chefmozaccepts)

new_df = userpayment.join(chefmozaccepts, "Upayment", "inner")

display(new_df)
