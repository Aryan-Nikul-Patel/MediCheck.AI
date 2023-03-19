import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Create a container for the heading
header_container = st.container()

# Add the hospital image and heading within the container
with header_container:
    # Create a column layout for the image and text
    col1, col2 = st.columns([1, 4])

    # Add the hospital image to the left column
    with col1:
        st.image('logo.png', width=150)

    # Add the heading to the right column
    with col2:
        st.markdown("<h1 style='text-align: center; color: red; font-size: 70px;'>MediCheck.AI</h1>", unsafe_allow_html=True)


# DIABETES
# loading the saved models
diabetes_model = pickle.load(open('saved models/diabetes_prediction.sav', 'rb'))
# loading the saved scaler
scaler_diabetes = pickle.load(open('scaler/scaler_diabetes.sav', 'rb'))

#HEART
heart_disease_model = pickle.load(open('saved models/heart_disease_prediction.sav','rb'))
scaler_heart = pickle.load(open('scaler/scaler_heart.sav', 'rb'))

#PARKINSON
parkinsons_model = pickle.load(open('saved models/Parkinsons_disease_prediction.sav', 'rb'))
scaler_parkinsons = pickle.load(open('scaler/scaler_parkinsons.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('MediCheck.AI : Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', value='1')
        SkinThickness = st.text_input('Skin Thickness value', value='29')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value='0.351')

    with col2:
       Glucose = st.text_input('Glucose Level', value='85')
       Insulin = st.text_input('Insulin Level', value='0')
       Age = st.text_input('Age of the Person', value='31')

    with col3:
       BloodPressure = st.text_input('Blood Pressure value', value='66')
       BMI = st.text_input('BMI value', value='26.6')

    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
         # change the input data to a numpy array
        input_data_as_numpy_array= np.asarray(input_data)

        # reshape the numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # performing scaling
        input_data_final = scaler_diabetes.transform(input_data_reshaped)
        diab_prediction = diabetes_model.predict(input_data_final)
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
     age = st.text_input('Age', value='58')
        
    with col2:
        sex = st.text_input('Sex', value='0')
            
    with col3:
        cp = st.text_input('Chest Pain types', value='0')
            
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', value='100')
            
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', value='248')
            
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl', value='0')
            
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results', value='0')
            
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved', value='122')
            
    with col3:
        exang = st.text_input('Exercise Induced Angina', value='0')
            
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', value='1.0')
            
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment', value='1')
            
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy', value='0')
            
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', value='2')

        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]

         # change the input data to a numpy array
        input_data_as_numpy_array= np.asarray(input_data)

        # reshape the numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # performing scaling
        input_data_final = scaler_heart.transform(input_data_reshaped)

        heart_prediction = heart_disease_model.predict(input_data_final)  
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
     fo = st.text_input('MDVP Fo(Hz)', value='122.400')
    
    with col2:
        fhi = st.text_input('MDVP Fhi(Hz)', value='148.650')
        
    with col3:
        flo = st.text_input('MDVP Flo(Hz)', value='113.819')
        
    with col4:
        Jitter_percent = st.text_input('MDVP Jitter(%)', value='0.00968')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP Jitter(Abs)', value='0.00008')
        
    with col1:
        RAP = st.text_input('MDVP RAP', value='0.00465')
        
    with col2:
        PPQ = st.text_input('MDVP PPQ', value='0.00696')
        
    with col3:
        DDP = st.text_input('Jitter DDP', value='0.01394')
        
    with col4:
        Shimmer = st.text_input('MDVP Shimmer', value='0.06134')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP Shimmer(dB)', value='0.626')
        
    with col1:
        APQ3 = st.text_input('Shimmer APQ3', value='0.03134')
        
    with col2:
        APQ5 = st.text_input('Shimmer APQ5', value='0.04518')
        
    with col3:
        APQ = st.text_input('MDVP APQ', value='0.04368')
        
    with col4:
        DDA = st.text_input('Shimmer DDA', value='0.09403')
        
    with col5:
        NHR = st.text_input('NHR', value='0.01929')
        
    with col1:
        HNR = st.text_input('HNR', value='19.085')
        
    with col2:
        RPDE = st.text_input('RPDE', value='0.458359')
        
    with col3:
        DFA = st.text_input('DFA', value='0.819521')
        
    with col4:
        spread1 = st.text_input('spread1', value='-4.075192')
        
    with col5:
        spread2 = st.text_input('spread2', value='0.335590')
        
    with col1:
        D2 = st.text_input('D2', value='2.48685')
        
    with col2:
        PPE = st.text_input('PPE', value='0.368674')

        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
        
         # change the input data to a numpy array
        input_data_as_numpy_array= np.asarray(input_data)

        # reshape the numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # performing scaling
        input_data_final = scaler_parkinsons.transform(input_data_reshaped)
        parkinsons_prediction = parkinsons_model.predict(input_data_final)
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

# # Create a container for the footer
# footer_container = st.container()

# with footer_container:
#     # Add the footer text
#     st.write("Made with ❤️ by Aryan Patel")

# Add custom CSS to position the footer at the bottom of the page
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a container for the footer
footer_container = st.container()

with footer_container:
    # Add the footer text and style it with the "footer" class
    st.write('<div class="footer">Made with ❤️ by Aryan Patel</div>', unsafe_allow_html=True)