import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.split(__file__)[0])

# Load models
model_dir = "./models_saved"
models = {
    "Logistic Regression": joblib.load(os.path.join(model_dir, "lr.joblib")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgb.joblib")),
    "Decision Tree": joblib.load(os.path.join(model_dir, "dt.joblib")),
    "SVM": joblib.load(os.path.join(model_dir, "svc.joblib")),
    "KNN": joblib.load(os.path.join(model_dir, "knn.joblib")),
    "Random Forest": joblib.load(os.path.join(model_dir, "rf.joblib"))
}

# Load scaler
scaler_path = os.path.join('./models_saved', 'scaler.joblib')
scaler = joblib.load(scaler_path)

# Define features and their default (mean) values
features = {
    'Initial Hb': 110.551020,
    'Initial PLT': 232.081633,
    'Time to start surgery after injury': 341.330612,
    'Duration of surgery': 167.236735,
    'TBSA grafted': 5.336735,
    'TBSA excised': 18.391837,
    'TBSA excised and grafted': 23.728571,
    'Amount of RBCs transfused during surgery': 681.020408
}

# Title
st.title("Machine learning model for predicting intraoperative blood loss > 750 ml during burn surgery")

# Sidebar inputs for features
st.sidebar.header("Input Features")

user_input = {}
for feature, mean_value in features.items():
    user_input[feature] = st.sidebar.number_input(f"{feature}:", value=mean_value, min_value=0.0, step=0.1)

# Select model
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Predict button
if st.sidebar.button("Predict"):
    # Convert user input to dataframe
    input_df = pd.DataFrame([user_input])

    # Standardize input
    input_scaled = scaler.transform(input_df)

    # Predict using the selected model
    model = models[model_name]
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    prediction = model.predict(input_scaled)[0]

    # Display prediction
    st.subheader(f"Prediction using {model_name}")
    st.write(f"Predicted risk of intraoperative blood loss > 750 ml:  **{prediction_proba:.2f}**")

    if prediction == 1:
        st.error("Predicted Class: High-risk  (Class 1)")
    else:
        st.success("Predicted class: Low-risk (Class 0)")

    # Display recommendations
    st.subheader("Recommendations")
    if prediction == 1:
        st.error("""
        **High Risk:**
        There is a significant risk of high blood loss during surgery. This scenario demands a multi-faceted approach to ensure patient safety and optimize outcomes. Here are several key recommendations:
    
        1. **Prepare for Potential Blood Transfusions:**
           - **Blood Bank Coordination:** Ensure that the blood bank is aware of the potential need for transfusions and that adequate blood products, such as packed red blood cells, fresh frozen plasma, and platelets, are available.
           - **Cross-Matching:** Perform cross-matching well in advance to prevent delays in case transfusions are required during surgery.
           - **Transfusion Protocols:** Establish clear protocols for when to initiate transfusions, including triggers such as hemoglobin levels, ongoing blood loss, and hemodynamic stability.
    
        2. **Ensure the Availability of Additional Blood Products:**
           - **Volume Expanders:** In addition to blood products, consider the availability of volume expanders such as crystalloids and colloids to manage acute blood loss.
           - **Specialized Products:** For patients with specific conditions (e.g., clotting disorders), ensure that specialized blood products, such as cryoprecipitate or factor concentrates, are readily available.
           - **Autologous Blood Salvage:** Consider the use of intraoperative blood salvage (cell saver) techniques to collect and reinfuse the patient's own blood, thereby reducing the need for allogeneic transfusions.
    
        3. **Advanced Surgical Techniques:**
           - **Hemostatic Measures:** Employ advanced hemostatic techniques, such as the use of topical hemostatic agents (e.g., fibrin sealants) and meticulous surgical technique to control bleeding at the source.
           - **Minimally Invasive Surgery:** Where feasible, consider minimally invasive surgical approaches (e.g., laparoscopic or robotic surgery) that are associated with reduced blood loss compared to open surgery.
           - **Vascular Control:** Ensure rapid access to vascular control measures, such as tourniquets, vessel loops, or temporary vascular shunts, to manage intraoperative hemorrhage.
    
        4. **Close Monitoring and Rapid Response:**
           - **Hemodynamic Monitoring:** Utilize continuous hemodynamic monitoring (e.g., arterial line) to detect and respond to changes in blood pressure and heart rate, which may indicate ongoing blood loss.
           - **Laboratory Monitoring:** Monitor serial hemoglobin and hematocrit levels during surgery to assess the extent of blood loss and guide transfusion decisions.
           - **Point-of-Care Testing:** Implement point-of-care testing (e.g., thromboelastography) to evaluate coagulation status in real-time, allowing for targeted management of coagulopathies.
    
        5. **Postoperative Care and Vigilance:**
           - **Intensive Care Unit (ICU) Monitoring:** Patients at high risk of significant blood loss may benefit from postoperative care in an ICU setting where close monitoring and rapid intervention are possible.
           - **Reevaluation:** Conduct frequent reassessments in the immediate postoperative period to ensure that any delayed bleeding or hemodynamic instability is promptly identified and managed.
           - **Nutritional Support:** Provide nutritional support, including iron supplementation, to facilitate recovery from anemia and to replenish lost blood components.
    
        6. **Patient and Family Communication:**
           - **Risk Communication:** Clearly communicate the risks of significant blood loss to the patient and their family, including the potential need for transfusions and the associated risks.
           - **Informed Consent:** Ensure that the informed consent process includes a detailed discussion of the potential for significant blood loss and the strategies in place to manage this risk.
    
        This comprehensive approach aims to minimize the risks associated with high blood loss during surgery and to ensure that all necessary resources are available to manage this challenging clinical situation effectively.
        """)
    
    else:
        st.success("""
        **Low Risk:**
        The risk of significant blood loss during surgery is assessed to be low, allowing for a more routine approach to patient management. However, standard precautions are essential to ensure patient safety. The following recommendations are provided:
    
        1. **Continue with Standard Surgical Protocols:**
           - **Routine Hemostasis:** Adhere to established surgical protocols for achieving hemostasis, including the use of electrocautery, sutures, and clamps to control minor bleeding.
           - **Surgical Planning:** Ensure that the surgical team has a clear plan in place, including preoperative assessment, intraoperative management, and postoperative care strategies.
    
        2. **Monitor Blood Loss as Per Standard Guidelines:**
           - **Intraoperative Assessment:** Monitor blood loss throughout the procedure using standard measures, such as visual estimation, weighing sponges, or using suction canisters.
           - **Hemodynamic Stability:** Regularly assess the patient’s hemodynamic stability during surgery to detect any unexpected blood loss early.
           - **Laboratory Monitoring:** Perform routine intraoperative laboratory tests, such as hemoglobin and hematocrit measurements, to ensure that blood loss remains within expected limits.
    
        3. **Be Prepared for Unexpected Situations:**
           - **Contingency Planning:** While the risk is low, it’s prudent to have contingency plans in place for unexpected situations that could lead to significant blood loss. This includes ensuring that the necessary equipment and personnel are available to manage such events.
           - **Staff Readiness:** The surgical team should remain vigilant and prepared to respond promptly to any unforeseen complications, including activating a rapid transfusion protocol if necessary.
    
        4. **Postoperative Care:**
           - **Routine Recovery:** Patients with low blood loss risk can typically be managed in a standard postoperative care setting, with routine monitoring of vital signs and clinical status.
           - **Early Mobilization:** Encourage early mobilization post-surgery to reduce the risk of thromboembolic events and to promote overall recovery.
           - **Nutritional Support:** Provide nutritional support to promote healing, particularly focusing on maintaining adequate iron levels and supporting erythropoiesis.
    
        5. **Patient and Family Communication:**
           - **Reassurance:** Reassure the patient and their family that the risk of significant blood loss is low and that the surgical team is well-prepared to handle the procedure safely.
           - **Postoperative Instructions:** Provide clear instructions regarding postoperative care, including signs of potential complications that should prompt immediate medical attention, such as unexpected bleeding or symptoms of anemia.
    
        By following these standard precautions, the surgical team can ensure that the procedure is conducted safely, minimizing the risk of complications while promoting a smooth recovery for the patient.
        """)
