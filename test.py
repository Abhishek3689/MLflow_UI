import os
import sys
import mlflow   
import mlflow.sklearn
from mlflow import log_metric,log_param,log_artifact
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


df=pd.read_csv("https://github.com/Abhishek3689/Test_Train_Datsets_CSV_Excel/raw/main/tips.csv")
df['sex']=df['sex'].astype('category')
df['smoker']=df['smoker'].astype('category')
df['day']=df['day'].astype('category')
df['time']=df['time'].astype('category')
#print(df.head())
ohe=OneHotEncoder()
df_cat=df[['sex','smoker','day','time']]
df_new=pd.get_dummies(df,columns=df_cat.columns,dtype=int)

X=df_new.drop('tip',axis=1)
y=df_new['tip']
X_train,X_test,y_train,y_test=train_test_split(X,y)

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train, y_train)

    predicted_qualities = lr.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    remote_server_uri = "https://dagshub.com/Abhishek3689/MLflow_UI.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetTipsModel")
    else:
        mlflow.sklearn.log_model(lr, "model")



# if __name__=="__main__":
#     # log a param
#     log_param("param1",5)

#     # log a metric
#     log_metric("foo",1)
#     log_metric("foo",2)
#     log_metric("foo",3)

#     # log an artifact
#     with open("outut.txt",'w') as f:
#         f.write("Writing Succes")
#     log_artifact('output.txt')
