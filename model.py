import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file_path = "E:\\Recommendation_system\\NIRF-Recommendation-System\\Dataset\\NIRF_Final.xlsx"

def load_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")  # Specify the engine
    return df

df = load_data(file_path)
print(df.columns.to_list())  # Print column names

#  'Institute ID'
score_columns = ['TLR(100)', 'RPC(100)', 'GO(100)', 'OI(100)', 'Perception(100)', 'Score', 'Ranking', 'TLR_SS_NT', 'TLR_SS_NE', 'TLR_SS_NP', 'TLR_SS_f(NT,NE)',
                 'TLR_SSf(NP)', 'TLR_SS', 'TLR_FSR_F', 'TLR_FSR_N (NT+NP)', 'TLR_FSR', 'TLR_FQE_FRA', 'TLR_FRA_Faculty_with_PhD', 'TLR_FQE_FQ', 'TLR_Total_no_of_Faculty', 'TLR_Faculty_with_exp_upto_8yrs',
                 'TLR_Faculty_with_exp_between_8_to_15yrs', 'TLR_Faculty_with_exp_above_15yrs', 'TLR_FQE_F1', 'TLR_FQE_FE', 'TLR_FQE_F2', 'TLR_FQE_F3', 'TLR_FQE', 'TLR_FRU_BC', 'TLR_FRU_BO',
                 'TLR_FRU', 'TLR_OE_RD', 'TLR_OE_GD', 'TLR_OE_ESCS', 'TLR_OE_DA', 'TLR_MR_Employer_Score', 'TLR_MR_Academic_Peer_Score', 'TLR_MR_Public_Perception_Score', 'TLR_OE_Score',
                 'TLR_MR_Score ', 'TLR_OE_MR_Final_Score ', 'RP_SDG4', 'RP_SDG5', 'RP_SDG6', 'RP_SDG7', 'RP_SDG13', 'RP_SDG9', 'RP_SDG', 'RP_PU_P','RP_FRQ_Required', 'RP_FRQ_available', 'RP_FRQ_min(Req,Aval)',
                 'RP_PU_f(P/FRQ)', 'RP_PU', 'RP_QP_CC', 'RP_QP_TOP25P', 'RP_QP_FRQ', 'RP_QP_f(CC/FRQ)', 'RP_QP-F( TOP25P/P)', 'RP_QP', 'RP_IPR_PG', 'RP_IPR_PP', 'RP_IPR', 'RP_FPPP_RF', 'RP_RF_Total_Research_Funding',
                 'RP_FPPP_CF', 'RP_CF_Total_Consultancy_Amt', 'RP_FPPP_FPC', 'RP_FPPP_FPR', 'RP_FPPP', 'GO_GPH_Np', 'GO_NP_PLACED_STD', 'GO_NP_Total_GRADUATING_STD', 'GO_GPH_Nhs','GO_NHS_STUDENTS_PLACED_FOR_HIGHER_STUDIES',
                 'GO_GPH', 'GO_GUE_Ng', 'GO_GUE', 'GO_GMS_MS', 'GO_GMS', 'GO_GPHD_Nphd', 'GO_GPHD', 'OI_RD_Students_Other_States', 'OI_RD_FractionOtherStates', 'OI_RD_Students_Other_Countries',
                 'OI_RD_FractionOtherCountries', 'OI_RD', 'OI_Female_Students', 'OI_Female_Faculty', 'OI_WD_NWS', 'OI_WD_NWF', 'OI_WD', 'OI_PCS', 'OI_ESCS_Nesc', 'OI_ESCS', 'PR_Employer_Score',
                 'PR_Academic_Peer_Score','PR_Accreditation_Score', 'PR_PeerPerception', 'PR_PR+Accr']

 # Count number of features
num_features = len(score_columns)
print(f"Total number of features: {num_features}")
# def normalize_scores(df, score_columns):

#     scaler = MinMaxScaler()
#     df[score_columns] = scaler.fit_transform(df[score_columns])
#     return df

from sklearn.impute import KNNImputer

df = load_data(file_path)
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
df_imputed = df.copy()
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print("Before Imputation: Missing Values Count\n", df.isnull().sum())
print("\nAfter KNN Imputation: Missing Values Count\n", df_imputed.isnull().sum())


df_imputed.to_csv(file_path, index=False)

"""Normalize"""

print(df[['Institute ID']].head(10))
print(df['Institute ID'].dtype)

df['Institute ID'] = df['Institute ID'].astype(str)

print(df[['Institute ID']].head(10))
print(df['Institute ID'].dtype)

from sklearn.preprocessing import MinMaxScaler
ranking_features = score_columns
scaler = MinMaxScaler()
df[ranking_features] = scaler.fit_transform(df[ranking_features])

from sklearn.metrics.pairwise import cosine_similarity

cosin_sim = cosine_similarity(df[ranking_features])

print("Cosine Similarity Matrix Shape:", cosin_sim.shape)
print(cosin_sim[:5, :5])

def get_similar_colleges(institute_id, n_neighbors=5):
    try:
        df['Institute ID'] = df['Institute ID'].astype(str)
        institute_id = str(institute_id)
        matching_rows = df[df['Institute ID'] == institute_id]
        if matching_rows.empty:
            print(f"Error: Institute ID {institute_id} not found in the dataset.")
            return []

        index = matching_rows.index[0]
        print(f"Institute ID {institute_id} found at index {index}")

        similarity_scores = list(enumerate(cosin_sim[index]))

        sorted_similarities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        print(f"Sorted Similarities: {sorted_similarities[:10]}")

        top_similar_colleges = df.iloc[[i[0] for i in sorted_similarities[1:n_neighbors+1]]][['Institute ID']]
        return top_similar_colleges

    except Exception as e:
        print(f"Error finding similar colleges: {str(e)}")
        return []

institute_id = "1.0"
top_colleges = get_similar_colleges(institute_id, n_neighbors=5)
print("Top Similar Colleges:\n", top_colleges)

print(df[df['Institute ID'] == "1.0"])

print("Unique Institute IDs in Dataset:\n", df['Institute ID'].unique())

print(df[['Institute ID']].head(10))
print(df['Institute ID'].dtype)

ranking_features_main = ['TLR_SS_NT', 'TLR_SS_NE', 'TLR_SS_NP', 'TLR_SS_f(NT,NE)',
                 'TLR_SSf(NP)', 'TLR_FSR_F', 'TLR_FSR_N (NT+NP)','TLR_FQE_FRA', 'TLR_FRA_Faculty_with_PhD', 'TLR_FQE_FQ', 'TLR_Total_no_of_Faculty', 'TLR_Faculty_with_exp_upto_8yrs',
                 'TLR_Faculty_with_exp_between_8_to_15yrs', 'TLR_Faculty_with_exp_above_15yrs', 'TLR_FQE_F1', 'TLR_FQE_FE', 'TLR_FQE_F2', 'TLR_FQE_F3','TLR_FRU_BC', 'TLR_FRU_BO',
                 'TLR_OE_RD', 'TLR_OE_GD', 'TLR_OE_ESCS', 'TLR_OE_DA',  'TLR_MR_Employer_Score', 'TLR_MR_Academic_Peer_Score', 'TLR_MR_Public_Perception_Score',
                  'RP_SDG4', 'RP_SDG5', 'RP_SDG6', 'RP_SDG7', 'RP_SDG13', 'RP_SDG9', 'RP_PU_P','RP_FRQ_Required', 'RP_FRQ_available', 'RP_FRQ_min(Req,Aval)',
                 'RP_PU_f(P/FRQ)', 'RP_QP_CC', 'RP_QP_TOP25P', 'RP_QP_FRQ', 'RP_QP_f(CC/FRQ)', 'RP_QP-F( TOP25P/P)','RP_IPR_PG', 'RP_IPR_PP',   'RP_FPPP_RF',
                  'RP_RF_Total_Research_Funding','RP_FPPP_CF', 'RP_CF_Total_Consultancy_Amt', 'RP_FPPP_FPC', 'RP_FPPP_FPR',  'GO_GPH_Np', 'GO_NP_PLACED_STD', 'GO_NP_Total_GRADUATING_STD',
                         'GO_GPH_Nhs','GO_NHS_STUDENTS_PLACED_FOR_HIGHER_STUDIES',  'GO_GUE_Ng',  'GO_GMS_MS', 'GO_GPHD_Nphd',
                          'OI_RD_Students_Other_States', 'OI_RD_FractionOtherStates', 'OI_RD_Students_Other_Countries',
                 'OI_RD_FractionOtherCountries', 'OI_Female_Students', 'OI_Female_Faculty', 'OI_WD_NWS', 'OI_WD_NWF', 'OI_ESCS_Nesc','PR_Employer_Score',
                 'PR_Academic_Peer_Score','PR_Accreditation_Score', 'PR_PeerPerception',]

target_features = ['TLR(100)', 'RPC(100)', 'GO(100)', 'OI(100)', 'Perception(100)', 'Score', 'Ranking',
                   'TLR_SS', 'TLR_FSR','TLR_FQE', 'TLR_FRU', 'TLR_OE_Score', 'TLR_MR_Score ','TLR_OE_MR_Final_Score ','RP_SDG', 'RP_PU'
                   ,'RP_QP', 'RP_IPR', 'RP_FPPP','GO_GPH',  'GO_GUE',
               'GO_GMS', 'GO_GPHD', 'OI_RD','OI_WD', 'OI_PCS', 'OI_ESCS',  'PR_PR+Accr'
                   ]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[ranking_features], df[target_features], test_size=0.2, random_state=42)

"""Rank Prediction"""

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

xgb_model = XGBRegressor(n_estimators = 100, learning_rate= 0.1, random_state=42)
xgb_model.fit(X_train, y_train)

y_prediction = xgb_model.predict(X_test)
print("XGBoost Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_prediction)))
print("R² Score:", r2_score(y_test, y_prediction))

import matplotlib.pyplot as plt

# Get feature importance
feature_importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': ranking_features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(100, 50))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Ranking Parameters")
plt.title("Feature Importance for NIRF Ranking")
plt.gca().invert_yaxis()
plt.show()

print(df[['Institute ID']].head(10))
print(df['Institute ID'].dtype)

def recommend_improvements(institute_id):
    try:

        institute_id = float(institute_id)

        #institute exist
        if institute_id not in df['Institute ID'].astype(float).values:
            return [f"Error: Institute ID {institute_id} not found in dataset."]


        institute_data = df[df['Institute ID'].astype(float) == institute_id][ranking_features].values[0]


        top_institutes = df[df['Ranking'] <= 10][ranking_features].mean()


        gaps = top_institutes - institute_data
        gaps_dict = dict(zip(ranking_features, gaps))


        sorted_gaps = sorted(gaps_dict.items(), key=lambda x: x[1], reverse=True)


        recommendations = []
        for feature, gap in sorted_gaps:
            if gap > 0.00:
                recommendations.append(f"Increase '{feature}' by at least {gap:.2f} points.")

        return recommendations if recommendations else ["Your institute is already performing well!"]

    except Exception as e:
        return [f"Error in recommendation: {str(e)}"]

institute_id = 1.0
improvement_suggestions = recommend_improvements(institute_id)
print("🔹 Recommended Improvements:")
for rec in improvement_suggestions:
    print("-", rec)
