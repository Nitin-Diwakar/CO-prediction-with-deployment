# from streamlit_pandas_profiling import st_profile_report
# from ydata_profiling import ProfileReport
# import ydata_profiling
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)


st.image("./image/pollution.jpg", width=500)
st.write('''
# Carbon Concentration prediction App
This web app will predict the **carbon concentration** in air
         ''')

st.write('---')


st.write("**Dataset Description**")
st.write("**CO:** True hourly averaged concentration CO in mg/m^3 (reference analyzer)")
st.write("**PT08.S1(CO):** PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)")
st.write("**C6H6(GT):** True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)")
st.write("**PT08.S2(NMHC):** PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)")
st.write("**NOx(GT):** True hourly averaged NOx concentration in ppb (reference analyzer)")
st.write("**PT08.S3(NOx):** PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)")
st.write("**NO2(GT):** True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)")
st.write("**PT08.S4(NO2):** PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)")
st.write("**PT08.S5(O3):** PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)")
st.write("**T:** Temperature in °C")
st.write("**RH:** Relative Humidity (%)")
st.write("**AH:** AH Absolute Humidity")

# Import dataset
df = pd.read_csv("air_quality_clean.csv")

# ===================== mean =============
pt_co_mean = df["PT08.S1(CO)"].mean()
c6h6_mean = df["C6H6(GT)"].mean()
pt_nhmc_mean = df["PT08.S2(NMHC)"].mean()
nox_mean = df["NOx(GT)"].mean()
pt_no_mean = df["PT08.S3(NOx)"].mean()
no2_mean = df["NO2(GT)"].mean()
pt_no2_mean = df["PT08.S4(NO2)"].mean()
pt_o3_mean = df["PT08.S5(O3)"].mean()
t_mean = df["T"].mean()
rh_mean = df["RH"].mean()
ah_mean = df["AH"].mean()


# ===================== standard deviation =============
pt_co_std = df["PT08.S1(CO)"].std()
c6h6_std = df["C6H6(GT)"].std()
pt_nhmc_std = df["PT08.S2(NMHC)"].std()
nox_std = df["NOx(GT)"].std()
pt_no_std = df["PT08.S3(NOx)"].std()
no2_std = df["NO2(GT)"].std()
pt_no2_std = df["PT08.S4(NO2)"].std()
pt_o3_std = df["PT08.S5(O3)"].std()
t_std = df["T"].std()
rh_std = df["RH"].std()
ah_std = df["AH"].std()


# view dataset

st.write(df)
st.markdown('<span style="color:red">*Note: Ignore comma(,) it just a streamlit issue</span>',
            unsafe_allow_html=True)

# ============= profiling report=================
# pr = df.profile_report()
# st_profile_report(pr)


# ========== Visualization======================
columns = list(df.columns)
st.sidebar.header('Plot setting')
char_selection = st.sidebar.selectbox(label='Type of plot',
                                      options=['Scatterplot', 'Boxplot'])

if char_selection == 'Scatterplot':
    st.sidebar.subheader("Scatterplot settings")
    x_value = st.sidebar.selectbox(label="X-axis",
                                   options=columns)
    y_value = st.sidebar.selectbox(label="Y-axis",
                                   options=columns)
    plot = px.scatter(data_frame=df, x=x_value, y=y_value)
    st.write(plot)

if char_selection == 'Boxplot':
    st.sidebar.subheader("Boxplot settings")
    value = st.sidebar.selectbox(label="Select Column",
                                 options=columns)
    plot = px.box(df, y=value)
    st.write(plot)


# slider for feature value selection
X = df.drop("CO(GT)", axis=1)
y = df["CO(GT)"]


# model selection ==========
st.sidebar.subheader("Model Selection")
select_model = st.sidebar.selectbox(label="Select model",
                                    options=["Linear", "Ridge", "Lasso", "ElasticNet"])


# =============== Linear regression=============
if select_model == "Linear":
    st.header("Specify Input Parameters")
    print(X.columns)

    def input_features():
        pt_co = st.slider(
            "**PT08.S1(CO)**", X["PT08.S1(CO)"].min(), X["PT08.S1(CO)"].max(), X["PT08.S1(CO)"].mean())
        c6h6 = st.slider(
            "**C6H6(GT)**", X["C6H6(GT)"].min(), X["C6H6(GT)"].max(), X["C6H6(GT)"].mean())
        pt_nhmc = st.slider('**PT08.S2(NMHC)**', X['PT08.S2(NMHC)'].min(
        ), X['PT08.S2(NMHC)'].max(), X['PT08.S2(NMHC)'].mean())
        nox = st.slider(
            '**NOx(GT)**', X['NOx(GT)'].min(), X['NOx(GT)'].max(), X['NOx(GT)'].mean())
        pt_no = st.slider(
            '**PT08.S3(NOx)**', X['PT08.S3(NOx)'].min(), X['PT08.S3(NOx)'].max(), X['PT08.S3(NOx)'].mean())
        no2 = st.slider(
            '**NO2(GT)**', X['NO2(GT)'].min(), X['NO2(GT)'].max(), X['NO2(GT)'].mean())
        pt_no2 = st.slider(
            '**PT08.S4(NO2)**', X['PT08.S4(NO2)'].min(), X['PT08.S4(NO2)'].max(), X['PT08.S4(NO2)'].mean())
        pt_o3 = st.slider(
            '**PT08.S5(O3)**', X['PT08.S5(O3)'].min(), X['PT08.S5(O3)'].max(), X['PT08.S5(O3)'].mean())
        t = st.slider('**T**', X['T'].min(), X['T'].max(), X['T'].mean())
        rh = st.slider('**RH**', X['RH'].min(), X['RH'].max(), X['RH'].mean())
        ah = st.slider('**AH**', X['AH'].min(), X['AH'].max(), X['AH'].mean())
        # ============== normalize input===================
        pt_co_norm = (pt_co - pt_co_mean)/pt_co_std
        c6h6_norm = (c6h6 - c6h6_mean) / c6h6_std
        pt_nhmc_norm = (pt_nhmc - pt_nhmc_mean) / pt_nhmc_std
        nox_norm = (nox - nox_mean) / nox_std
        pt_no_norm = (pt_no - pt_no_mean) / pt_no_std
        no2_norm = (no2 - no2_mean) / no2_std
        pt_no2_norm = (pt_no2 - pt_no2_mean) / pt_no2_std
        pt_o3_norm = (pt_o3 - pt_o3_mean) / pt_o3_std
        t_norm = (t - t_mean) / t_std
        rh_norm = (rh - rh_mean) / rh_std
        ah_norm = (ah - ah_mean) / ah_std

        data = {
            'PT08.S1(CO)': pt_co_norm,
            'C6H6(GT)': c6h6_norm,
            'PT08.S2(NMHC)': pt_nhmc_norm,
            'NOx(GT)': nox_norm,
            'PT08.S3(NOx)': pt_no_norm,
            'NO2(GT)': no2_norm,
            'PT08.S4(NO2)': pt_no2_norm,
            'PT08.S5(O3)': pt_o3_norm,
            'T': t_norm,
            'RH': rh_norm,
            'AH': ah_norm
        }

        feature = pd.DataFrame([data])

        return feature

    x = input_features()
    st.write(x)

    st.write("Predicted CO")

    model = pickle.load(open('model.sav', 'rb'))

    pred = str(round(model.predict(x)[0], 4))
    res = "The CO(GT) of input feature is " + pred
    st.success(res)


# ================= Redge regression=============
if select_model == "Ridge":
    st.header("Specify Input Parameters")
    print(X.columns)

    def input_features():
        pt_co = st.slider(
            "**PT08.S1(CO)**", X["PT08.S1(CO)"].min(), X["PT08.S1(CO)"].max(), X["PT08.S1(CO)"].mean())
        c6h6 = st.slider(
            "**C6H6(GT)**", X["C6H6(GT)"].min(), X["C6H6(GT)"].max(), X["C6H6(GT)"].mean())
        pt_nhmc = st.slider('**PT08.S2(NMHC)**', X['PT08.S2(NMHC)'].min(
        ), X['PT08.S2(NMHC)'].max(), X['PT08.S2(NMHC)'].mean())
        nox = st.slider(
            '**NOx(GT)**', X['NOx(GT)'].min(), X['NOx(GT)'].max(), X['NOx(GT)'].mean())
        pt_no = st.slider(
            '**PT08.S3(NOx)**', X['PT08.S3(NOx)'].min(), X['PT08.S3(NOx)'].max(), X['PT08.S3(NOx)'].mean())
        no2 = st.slider(
            '**NO2(GT)**', X['NO2(GT)'].min(), X['NO2(GT)'].max(), X['NO2(GT)'].mean())
        pt_no2 = st.slider(
            '**PT08.S4(NO2)**', X['PT08.S4(NO2)'].min(), X['PT08.S4(NO2)'].max(), X['PT08.S4(NO2)'].mean())
        pt_o3 = st.slider(
            '**PT08.S5(O3)**', X['PT08.S5(O3)'].min(), X['PT08.S5(O3)'].max(), X['PT08.S5(O3)'].mean())
        t = st.slider('**T**', X['T'].min(), X['T'].max(), X['T'].mean())
        rh = st.slider('**RH**', X['RH'].min(), X['RH'].max(), X['RH'].mean())
        ah = st.slider('**AH**', X['AH'].min(), X['AH'].max(), X['AH'].mean())

        # ============== normalize input===================
        pt_co_norm = (pt_co - pt_co_mean)/pt_co_std
        c6h6_norm = (c6h6 - c6h6_mean) / c6h6_std
        pt_nhmc_norm = (pt_nhmc - pt_nhmc_mean) / pt_nhmc_std
        nox_norm = (nox - nox_mean) / nox_std
        pt_no_norm = (pt_no - pt_no_mean) / pt_no_std
        no2_norm = (no2 - no2_mean) / no2_std
        pt_no2_norm = (pt_no2 - pt_no2_mean) / pt_no2_std
        pt_o3_norm = (pt_o3 - pt_o3_mean) / pt_o3_std
        t_norm = (t - t_mean) / t_std
        rh_norm = (rh - rh_mean) / rh_std
        ah_norm = (ah - ah_mean) / ah_std

        data = {
            'PT08.S1(CO)': pt_co_norm,
            'C6H6(GT)': c6h6_norm,
            'PT08.S2(NMHC)': pt_nhmc_norm,
            'NOx(GT)': nox_norm,
            'PT08.S3(NOx)': pt_no_norm,
            'NO2(GT)': no2_norm,
            'PT08.S4(NO2)': pt_no2_norm,
            'PT08.S5(O3)': pt_o3_norm,
            'T': t_norm,
            'RH': rh_norm,
            'AH': ah_norm
        }
        feature = pd.DataFrame([data])
        # scale = StandardScaler()
        # col = list(feature.columns)
        # feature.loc[:, col] = scale.fit_transform(feature[col])

        return feature

    x = input_features()
    st.write(x)

    st.write("Predicted CO")

    model = pickle.load(open('reg.sav', 'rb'))

    pred = str(round(model.predict(x)[0], 4))
    res = "The CO(GT) of input feature is " + pred
    st.success(res)

if select_model == "Lasso":
    st.header("Work in Progress")
if select_model == "ElasticNet":
    st.header("Work in Progress")
