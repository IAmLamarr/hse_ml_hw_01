import json
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pandas import DataFrame
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


X_train = pd.read_csv("cars_train.csv")
X_train_cat = pickle.load(open("X_train_cat.pickle", "rb"))
scaler = StandardScaler()
scaler.fit(X_train_cat)
encoder = pickle.load(open("encoder.pickle", "rb"))
model = pickle.load(open("model.pickle", "rb"))


def cast_float(elem):
    if type(elem) == float:
        return elem
    else:
        float_part = elem.split(" ")[0]
        if float_part == "":
            return 0
        return float(elem.split(" ")[0])


def encode_cat(dataframe):
    cat_columns = ["fuel", "seller_type", "transmission", "owner", "seats"]

    dataset = dataframe.copy()
    new_cat_features = encoder.get_feature_names_out()
    new_features = encoder.transform(dataset[cat_columns])
    dataset[new_cat_features] = new_features
    dataset = dataset.drop(cat_columns, axis=1)
    return dataset


def convert_items_to_df(items: List[Item]):
    model_fields = {name: info.annotation for name, info in Item.model_fields.items()}
    field_names = list(model_fields.keys())

    values = []
    for item in items:
        values.append(list(item.model_dump().values()))

    values = np.array(values).reshape(len(items), -1)

    df = pd.DataFrame(values, columns=field_names)
    df = df.astype(dtype=model_fields)
    df = df.drop("selling_price", axis=1)
    return df


def prepare_items(df: DataFrame):
    df = df.drop(["name", "torque"], axis=1)

    nan_columns = df.columns[df.isna().any()].tolist()

    float_features = ["mileage", "engine", "max_power"]
    for column in float_features:
        df[column] = df[column].apply(cast_float)

    for nan_column in nan_columns:
        median = df[nan_column].median()
        df[nan_column] = df[nan_column].fillna(median)

    df["engine"] = df["engine"].astype(int)
    df["seats"] = df["seats"].astype(int).astype(str)

    df = encode_cat(df)

    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    return df_scaled


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    dataframe = convert_items_to_df([item])
    prepared_item = prepare_items(dataframe)
    return model.predict(prepared_item)[0]


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    dataframe = convert_items_to_df(items.objects)
    prepared_items = prepare_items(dataframe)
    return model.predict(prepared_items)


@app.post("/predict_items_csv")
def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    selling_price = df.pop("selling_price")
    prepared_items = prepare_items(df)
    predicted = model.predict(prepared_items)
    df["selling_price"] = selling_price
    df["predicted"] = predicted
    return json.loads(df.to_json(orient="records"))
