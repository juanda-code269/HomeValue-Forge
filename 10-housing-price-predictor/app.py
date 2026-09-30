import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso,LinearRegression,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

st.set_page_config(page_title="HomeValue Lab",page_icon="🏠",layout="wide")
st.title("HomeValue Lab — regression and uncertainty")
st.warning("Educational model estimate only; this is not a professional appraisal or valuation.")


@st.cache_data
def demo(n=3500):
    rng=np.random.default_rng(15);location=rng.choice(["Urban","Suburban","Rural","Coastal"],n,p=[.32,.38,.2,.1])
    sqft=np.exp(rng.normal(7.35,.42,n)).clip(450,6500);beds=np.clip(np.round(sqft/620+rng.normal(0,.8,n)),1,8).astype(int)
    baths=np.clip(np.round(beds*.65+rng.normal(0,.55,n),1),1,6);year=rng.integers(1940,2026,n);lot=np.exp(rng.normal(8.7,.8,n)).clip(900,150000)
    loc={"Urban":110000,"Suburban":65000,"Rural":-35000,"Coastal":240000}
    price=45000+185*sqft+21000*baths+np.array([loc[x] for x in location])+450*(year-1980)+.35*lot-0.012*np.maximum(sqft-3600,0)**2+rng.normal(0,65000,n)
    return pd.DataFrame({"square_feet":sqft,"bedrooms":beds,"bathrooms":baths,"year_built":year,"lot_size":lot,"location":location,"price":price.clip(60000)})


def load(upload):
    if upload is None:return demo(),"Synthetic housing demonstration data"
    d=pd.read_csv(upload);target=next((c for c in d if c.lower() in {"price","saleprice","sale_price","target"}),None)
    if not target:raise ValueError("CSV needs a price/sale_price target column.")
    return d.rename(columns={target:"price"}),f"Uploaded data: {upload.name}"


upload=st.sidebar.file_uploader("Optional housing CSV",type="csv")
try:
    df,source=load(upload);df=df.dropna(subset=["price"])
    if len(df)<200:raise ValueError("At least 200 rows are required.")
except Exception as exc:st.error(str(exc));st.stop()
X=df.drop(columns="price");y=df.price;nums=X.select_dtypes(include=np.number).columns.tolist();cats=[c for c in X if c not in nums]
prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)])
idx_train,idx_test=train_test_split(np.arange(len(df)),test_size=.25,random_state=42)
models={"Linear":LinearRegression(),"Ridge":Ridge(alpha=10),"Lasso":Lasso(alpha=100,max_iter=5000),"Random Forest":RandomForestRegressor(n_estimators=220,min_samples_leaf=4,n_jobs=-1,random_state=42),"Gradient Boosting":GradientBoostingRegressor(n_estimators=180,max_depth=2,loss="huber",random_state=42)}
fitted={};rows=[]
for name,m in models.items():
    pipe=Pipeline([("prep",prep),("model",m)]).fit(X.iloc[idx_train],y.iloc[idx_train]);pred=pipe.predict(X.iloc[idx_test]);fitted[name]=(pipe,pred)
    rows.append({"Model":name,"MAE":mean_absolute_error(y.iloc[idx_test],pred),"RMSE":mean_squared_error(y.iloc[idx_test],pred)**.5,"R²":r2_score(y.iloc[idx_test],pred)})
choice=st.sidebar.selectbox("Active model",list(models),index=3);pipe,pred=fitted[choice]
tabs=st.tabs(["EDA","Models & residuals","Estimate a property","Importance","Limits"])
with tabs[0]:
    st.info(source);a,b,c=st.columns(3);a.metric("Homes",f"{len(df):,}");b.metric("Median price",f"${y.median():,.0f}");c.metric("Missing cells",f"{df.isna().sum().sum():,}")
    c1,c2=st.columns(2);c1.plotly_chart(px.histogram(df,x="price",nbins=45,marginal="box",title="Target distribution"),width="stretch")
    feature=c2.selectbox("Inspect numeric relationship",nums);c2.plotly_chart(px.scatter(df.sample(min(1600,len(df)),random_state=1),x=feature,y="price",trendline="lowess",title="Potential nonlinearity"),width="stretch")
    st.plotly_chart(px.imshow(df.select_dtypes(include=np.number).corr(),color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Numeric correlations"),width="stretch")
with tabs[1]:
    st.dataframe(pd.DataFrame(rows).style.format({"MAE":"${:,.0f}","RMSE":"${:,.0f}","R²":"{:.3f}"}),width="stretch")
    residual=y.iloc[idx_test].to_numpy()-pred;chart=pd.DataFrame({"Predicted":pred,"Residual":residual,"Actual":y.iloc[idx_test].to_numpy()})
    st.plotly_chart(px.scatter(chart,x="Predicted",y="Residual",color="Actual",title=f"Residuals — {choice}"),width="stretch")
    st.caption("MAE is the typical absolute miss; RMSE penalizes large misses more heavily.")
with tabs[2]:
    values={};cols=st.columns(2)
    for i,col in enumerate(X.columns):
        with cols[i%2]:
            if col in nums:
                values[col]=st.number_input(col,float(X[col].quantile(.01)),float(X[col].quantile(.99)),float(X[col].median()))
            else:values[col]=st.selectbox(col,sorted(X[col].dropna().astype(str).unique()))
    scenario=pd.DataFrame([values]);estimate=float(pipe.predict(scenario)[0]);resid=y.iloc[idx_test].to_numpy()-pred
    lo,hi=estimate+np.quantile(resid,[.1,.9]);st.metric("Model estimate",f"${estimate:,.0f}");st.write(f"Empirical 80% residual interval: **${max(0,lo):,.0f} – ${hi:,.0f}**")
    st.caption("This interval is a simple holdout-residual interval, not a guarantee and not necessarily valid for unusual properties.")
with tabs[3]:
    names=pipe.named_steps["prep"].get_feature_names_out();m=pipe.named_steps["model"]
    importance=np.abs(m.coef_) if hasattr(m,"coef_") else m.feature_importances_
    imp=pd.DataFrame({"Feature":names,"Importance":importance}).nlargest(20,"Importance").sort_values("Importance")
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Global feature influence"),width="stretch")
with tabs[4]:
    st.markdown("""Market conditions change, location is difficult to encode, omitted renovations and condition matter, and public records can be stale. Preprocessing is fit only on the training split to reduce leakage. Predictions outside the observed feature range are extrapolations. Geographic and price-range error analysis should be performed before any serious use; this educational app is intentionally not a valuation service.""")

