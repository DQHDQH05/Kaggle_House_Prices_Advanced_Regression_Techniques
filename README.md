# Kaggle-House-Prices---Advanced-Regression-Techniques
Kaggle project: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Let's do this together! Share any thoughts that you believe will build a better model!


################################################################################################## <br />
3/28/2022 <br />
################################################################################################## <br />

Learning objectives: 
1. Know how to clean data for supervised learning
2. Use 4-6 different models (Tree: lightGBM, XGBoost? Linear Regression: Ridge, Lasso, Elastic Net, PLS regression)
3. get RMSE as small as possible, ideally top 15%


3/28/2022 - 4/3/2022 <br />
Data Clean <br />
1) Check distribution, then determining outliers
2) Think about how to deal with NA and 0 values

Data clean job for each member: <br />
COL 1-20: Qu Zhou <br />
COL 21-40: Ziyi Li <br />
COL 41-60: Tianyu Ying <br />
COL 61-79: Qihong Dai <br />

4/4/2022 discussion for next step!

################################################################################################## <br />
4/5/2022 <br />
################################################################################################## <br />

Objective:
1. Keep all instances (rows)
2. Delete features if >50% NaN and NaN don't have meanings
3. KNN regression to fill resting NaN values based on other value of this feature (x=this feature, y=price) --> Use this as primary method (Plan B: use mean/median values)
4. Don't change categorical features to numerical ones, convert all categorical features to dummies (0,1) (e.g. Alley --> Alley_Grvl, Alley_Paved, Alley_NA)
