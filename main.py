import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer
from utils import ColumnSelector, TypeSelector, MyLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
train_csv = (pd.read_csv('./data/train.csv'))
train_csv['GarageQual'].fillna(value='No Garage', inplace=True)

#print(train_csv.isna().sum().sort_values(ascending=False))

cols = train_csv.columns
numeric_cols = train_csv._get_numeric_data().columns
categorical_cols= list(set(cols) - set(numeric_cols))
print(categorical_cols)

X = train_csv.drop(labels=['SalePrice'], axis=1)
y = train_csv['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#ejemplo de pipiline para categ features de Garage
categorical_pipleline =  make_pipeline(
    ColumnSelector(['GarageQual']),
    SimpleImputer(strategy='constant', fill_value='No Garage'),
    MyLabelBinarizer()
)
print(categorical_pipleline.fit_transform(X))


exit()

column_list = []
fill_value_list = []
column_list.append('LotFrontage')
fill_value_list.append(train_csv['LotFrontage'].mean())
train_csv['LotFrontage'].fillna(value=train_csv['LotFrontage'].mean(), inplace=True)

le = OrdinalEncoder()
train_csv['Alley'] = train_csv['Alley'].astype('category').cat.codes
train_csv['PoolQC'] = train_csv['PoolQC'].astype('category').cat.codes
train_csv['Fence'] = train_csv['Fence'].astype('category').cat.codes
train_csv['FireplaceQu'] = train_csv['FireplaceQu'].astype('category').cat.codes
train_csv['GarageType'] = train_csv['GarageType'].astype('category').cat.codes
train_csv['GarageCond'] = train_csv['GarageCond'].astype('category').cat.codes
train_csv['MasVnrType'] = train_csv['MasVnrType'].astype('category').cat.codes
train_csv['BsmtFinType2'] = train_csv['BsmtFinType2'].astype('category').cat.codes
train_csv['BsmtExposure'] = train_csv['BsmtExposure'].astype('category').cat.codes
train_csv['BsmtFinType1'] = train_csv['BsmtFinType1'].astype('category').cat.codes
train_csv['BsmtCond'] = train_csv['BsmtCond'].astype('category').cat.codes
train_csv['BsmtQual'] = train_csv['BsmtQual'].astype('category').cat.codes

column_list.append('MasVnrArea')
fill_value_list.append(train_csv['MasVnrArea'].mean())
train_csv['MasVnrArea'].fillna(value=train_csv['MasVnrArea'].mean(), inplace=True)

train_csv.drop(labels=['GarageFinish', 'GarageQual', 'GarageYrBlt', 'Electrical'], axis=1, inplace=True)


train_csv.drop(labels=['MiscFeature'], axis=1, inplace=True)

