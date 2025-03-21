import pandas as pd

def get_user_input(columns):
   


   
    data = {}
    for col in columns:
        try:
            data[col] = float(input(f"Enter value for {col}: "))
        except ValueError:
            print(f"Invalid input for {col}, setting to 0.")
            data[col] = 0
    return data

def calculate_metrics(df):
    df['TLR_SSf(NP)'] = df['TLR_SS_NP'] / df['TLR_SS_NP'].max()
    df['TLR_SS_f(NT,NE)'] = df['TLR_SS_NT'] / df['TLR_SS_NE']
    df['TLR_SS'] = df['TLR_SS_f(NT,NE)'] * 15 + df['TLR_SSf(NP)'] * 5
    
    df['TLR_FSR_N (NT+NP)'] = df['TLR_SS_NT'] + df['TLR_SS_NP']
    df['TLR_FSR'] = 30 * (15 * (df['TLR_FSR_F'] / df['TLR_FSR_N (NT+NP)']))
    
    df['TLR_FQE_FRA'] = (df['TLR_FRA_Faculty_with_PhD'] / df['TLR_Total_no_of_Faculty']).round(4)
    df['TLR_FQE_FQ'] = df['TLR_FQE_FRA'].apply(lambda FRA: 10 * (FRA / 95) if FRA < 95 else 10)
    df['TLR_FQE_FE'] = (3 * df['TLR_FQE_F1'].apply(lambda x: min(3 * x, 1)) +
                         3 * df['TLR_FQE_F2'].apply(lambda x: min(3 * x, 1)) +
                         4 * df['TLR_FQE_F3'].apply(lambda x: min(3 * x, 1)))
    df['TLR_FQE'] = df['TLR_FQE_FQ'] + df['TLR_FQE_FE']
    
    df['TLR_FRU'] = 7.5 * df['TLR_FRU_BC'] + 22.5 * df['TLR_FRU_BO']
    
    df['TLR_OE_Score'] = (df['TLR_OE_RD'] + df['TLR_OE_GD'] + df['TLR_OE_ESCS'] + df['TLR_OE_DA']) / 4
    df['TLR_MR_Score'] = (df['TLR_MR_Employer_Score'] * 0.4) + (df['TLR_MR_Academic_Peer_Score'] * 0.4) + (df['TLR_MR_Public_Perception_Score'] * 0.2)
    df['TLR_OE_MR_Final_Score'] = df['TLR_OE_Score'] * 0.5 + df['TLR_MR_Score']
    
    df['RP_PU_f(P/FRQ)'] = df['RP_PU_P'] / df['RP_PU_FRQ']
    df['RP_PU'] = 35 * df['RP_PU_f(P/FRQ)']
    
    df['RP_FRQ_Required'] = df['TLR_SS_NT'] / 15
    df['RP_FRQ_available'] = df['Total_no_of_Faculty']
    df['RP_FRQ_min(Req,Aval)'] = df[['RP_FRQ_Required', 'RP_FRQ_available']].min(axis=1)
    df['RP_QP_f(CC/FRQ)'] = df['RP_QP_CC'] / df['RP_QP_FRQ']
    df['RP_QP-F( TOP25P/P)'] = df['RP_QP_TOP25P'] / df['RP_PU_P']
    df['RP_QP'] = 20 * df['RP_QP_f(CC/FRQ)'] + 20 * df['RP_QP-F( TOP25P/P)']
    df['RP_IPR'] = (10 * df['RP_IPR_PG']) + (5 * df['RP_IPR_PP'])
    df['RP_FPPP_RF'] = df['RP_RF_Total_Research_Funding'] / df['RP_Total_no_of_Faculty']
    df['RP_FPPP_CF'] = df['RP_CF_Total_Consultancy_Amt'] / df['RP_Total_no_of_Faculty']
    df['RP_FPPP_FPR'] = 7.5 * df['RP_FPPP_RF']
    df['RP_FPPP_FPC'] = 2.5 * df['RP_FPPP_CF']
    df['RP_FPPP'] = df['RP_FPPP_FPR'] + df['RP_FPPP_FPC']
    
    df['RP_SDG'] = (df['RP_SDG4'] * 0.25) + (df['RP_SDG5'] * 0.15) + (df['RP_SDG6'] * 0.15) + (df['RP_SDG7'] * 0.15) + (df['RP_SDG13'] * 0.15) + (df['RP_SDG9'] * 0.15)
    
    df['GO_GPH_Np'] = (df['GO_NP_PLACED_STD'] / df['GO_NP_Total_GRADUATING_STD']) * 100
    df['GO_GPH_Nhs'] = (df['GO_NHS_STUDENTS_PLACED_FOR_HIGHER_STUDIES'] / df['GO_NP_Total_GRADUATING_STD']) * 100
    df['GPH'] = 40 * ((df['GO_GPH_Np'] / 100) + (df['GO_GPH_Nhs'] / 100))
    df['GUE'] = 15 * df['GO_GUE_Ng'].apply(lambda Ng: min(Ng / 80, 1))
    df['GMS'] = 25 * df['GO_GMS_MS']
    df['GO_GPHD'] = 20 * df['GO_GPHD_Nphd']
    
    df['OI_RD'] = 25 * df['OI_RD_FractionOtherStates'] + 5 * df['OI_RD_FractionOtherCountries']
    df['OI_ESCS'] = 20 * df['OI_ESCS_Nesc']
    df['OI_WD'] = (df['OI_WD_NWS'] * 0.5) + (df['OI_WD_NWF'] * 0.5)
    df['PR_PR+Accr'] = (df['PR_Employer_Score'] * 0.4) + (df['PR_Academic_Peer_Score'] * 0.4) + (df['PR_Accreditation_Score'] * 0.2)
    return df

if __name__ == "__main__":
    columns = ['TLR_SS_NT', 'TLR_SS_NE', 'TLR_SS_NP', 'TLR_SS_f(NT,NE)',
                 'TLR_SSf(NP)', 'TLR_FSR_F', 'TLR_FSR_N (NT+NP)','TLR_FQE_FRA', 'TLR_FRA_Faculty_with_PhD', 'TLR_FQE_FQ', 'TLR_Total_no_of_Faculty', 'TLR_Faculty_with_exp_upto_8yrs',
                 'TLR_Faculty_with_exp_between_8_to_15yrs', 'TLR_Faculty_with_exp_above_15yrs', 'TLR_FQE_F1', 'TLR_FQE_FE', 'TLR_FQE_F2', 'TLR_FQE_F3','TLR_FRU_BC', 'TLR_FRU_BO',
                 'TLR_OE_RD', 'TLR_OE_GD', 'TLR_OE_ESCS', 'TLR_OE_DA',  'TLR_MR_Employer_Score', 'TLR_MR_Academic_Peer_Score', 'TLR_MR_Public_Perception_Score',
                  'RP_SDG4', 'RP_SDG5', 'RP_SDG6', 'RP_SDG7', 'RP_SDG13', 'RP_SDG9', 'RP_PU_P','RP_FRQ_Required', 'RP_FRQ_available', 'RP_FRQ_min(Req,Aval)',
                 'RP_PU_f(P/FRQ)', 'RP_QP_CC', 'RP_QP_TOP25P', 'RP_QP_FRQ', 'RP_QP_f(CC/FRQ)', 'RP_QP-F( TOP25P/P)','RP_IPR_PG', 'RP_IPR_PP',   'RP_FPPP_RF',
                  'RP_RF_Total_Research_Funding','RP_FPPP_CF', 'RP_CF_Total_Consultancy_Amt', 'RP_FPPP_FPC', 'RP_FPPP_FPR',  'GO_GPH_Np', 'GO_NP_PLACED_STD', 'GO_NP_Total_GRADUATING_STD',
                         'GO_GPH_Nhs','GO_NHS_STUDENTS_PLACED_FOR_HIGHER_STUDIES',  'GO_GUE_Ng',  'GO_GMS_MS', 'GO_GPHD_Nphd',
                          'OI_RD_Students_Other_States', 'OI_RD_FractionOtherStates', 'OI_RD_Students_Other_Countries',
                 'OI_RD_FractionOtherCountries', 'OI_Female_Students', 'OI_Female_Faculty', 'OI_WD_NWS', 'OI_WD_NWF', 'OI_ESCS_Nesc','PR_Employer_Score',
                 'PR_Academic_Peer_Score','PR_Accreditation_Score', 'PR_PeerPerception']
    user_data = get_user_input(columns)
    df = pd.DataFrame([user_data])
    df = calculate_metrics(df)
    print(df)
