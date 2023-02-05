import streamlit as st
import numpy as np
import streamlit.components.v1 as components
from constant import *
import webbrowser
from PIL import Image
import base64
from pathlib import Path
from torchvision.transforms import transforms
import GAN
import pneumonia_classification_model
import pandas as pd


#TODO Create a new file for all helper functions for now paste it down here
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Function to read and manipulate images
def load_img(img):
    im = Image.open(img)
    image = np.array(im)
    return image

with st.sidebar:

    #TODO Add Picture here
    st.markdown("<h1>KETUL POLARA</h1>", unsafe_allow_html=True)

    pages = ["ABOUT ME", "EDUCATION", "EXPERIENCE", "PROJECTS"]
    section = st.sidebar.radio('', pages)  # this is my sidebar radio button widget

    # hidden div with anchor
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)

if section == 'ABOUT ME':
    # TODO thing about this description since you copied from sagar
    st.markdown("<h2>ABOUT ME</h2>", unsafe_allow_html=True)
    picture, content = st.columns([1, 1])
    with picture:
        header_html_my_picture = "<img src='data:image/png;base64,{}' class='img-fluid' width='234.25' height='280.5' style='display: block;margin-top: 10px'>".format(
            img_to_bytes("images/1.jpg")
        )
        st.markdown(
            header_html_my_picture, unsafe_allow_html=True,
        )
    with content:
        st.write(
            "I am curiosity driven Data Scientist/Software Engineer with a demonstrated ability to deliver meaningful insights, make informed decisions, and solve challenging business problems by leveraging Statistics and advanced data-driven methods. Moreover, I am competent in programming language proficiency and statistical model development with proficiency in research.")
        st.write("Name: Ketul Polara")
        st.write("Email: kpola009@fiu.edu")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
                   <a href="https://github.com/kpola009" target = "_blank"> 
                       <button style="background-color:GreenYellow;">Github</button> 
                   </a>
               """, unsafe_allow_html=True)

    # TODO Add a function to download your resume
    with col2:
        st.button('Resume')

    components.html(embed_component['linkedin'], height=310)

if section == 'EDUCATION':
    ### EDUCATION Section
    st.markdown("<h2>EDUCATION</h2>", unsafe_allow_html=True)
    st.write("")
    education_col1, education_col2 = st.columns([1,2])

    header_html_master = "<img src='data:image/png;base64,{}' class='img-fluid' width='180' height='81' style='display: block;margin-top: 110px'>".format(
        img_to_bytes("images/fiu-alone.png")
    )

    header_html_bachelor = "<img src='data:image/png;base64,{}' class='img-fluid' width='180' height='81' style='display: block;margin-top: 110px'>".format(
        img_to_bytes("images/fiu-alone.png")
    )

    with education_col1:
        fiu_image = Image.open("images/fiu-alone.png")
        fiu_image = fiu_image.resize((180,81))
        st.markdown(
            header_html_master, unsafe_allow_html=True,
        )

    with education_col2:
        st.markdown("<h2>Master of Science in Information Technology</h2>", unsafe_allow_html=True)
        st.markdown("<h4>Florida International University</h4>", unsafe_allow_html=True)
        st.write("August 2021 - April 2023")
        st.write("Relevant Coursework: Principals of Data Mining, Advance Data Mining, Advance Sensor/IoT Deep Learning, Advance Special Topics (3), Advance Software Engineering, Software and Data Modeling, Operating Systems.")

    st.write("")
    st.write("")

    education_b_col1, education_b_col2 = st.columns([1,2])

    with education_b_col1:
        st.markdown(
            header_html_bachelor, unsafe_allow_html=True,
        )

    with education_b_col2:
        st.markdown("<h2>Bachelor of Science in Information Technology</h2>", unsafe_allow_html=True)
        st.markdown("<h4>Florida International University</h4>", unsafe_allow_html=True)
        st.write("January 2017 - December 2020")
        st.write("Relevant Coursework: Component Software Development, Intermediate Java Programming, Database Systems, Database Admin, Operating Systems, UNIX System Admin, Enterprise IT Troubleshoot, Web Application Programming, Website Construction and Management.")



### EXPERIENCE Section
if section == "EXPERIENCE":
    st.markdown("<h2>EXPERIENCE</h2>", unsafe_allow_html=True)
    st.write("")

    experience_col1, experience_col2 = st.columns([1,2])

    header_html_exp = "<img src='data:image/png;base64,{}' class='img-fluid' width='160'  style='display: block;margin-top: 150px'>".format(
        img_to_bytes("images/Energy_Systems_Lab_Logo_Final-e1472763643607.png")
    )
    with experience_col1:

        st.markdown(
            header_html_exp, unsafe_allow_html=True,
        )

    with experience_col2:
        st.markdown("<h2>Machine Learning Researcher (GRA)</h2>", unsafe_allow_html=True)
        st.markdown("<h4>Energy Systems Research Laboratory (FIU)</h4>", unsafe_allow_html=True)
        st.write("August 2021 - Current")
        st.markdown("<ul><li>Designed and developed Time-series database (InfluxDB) to capture data from 200 data points using RTI Data distributions service (Similar to Apache Kafka) for smart grid testbed.</li>"
                    "<li>Using AWS SageMaker, S3, and SageMaker endpoint, trained and deployed Autoencoder and Isolation Forest Machine Learning models, in python for anomaly detection in smart grid testbed.</li>"
                    "<li>Implemented Federated learning Framework Flower with ANN and CNN to detect DoS attacks on IoT devices.</li>"
                    "<li>Implemented Forecasting models for Load and Solar Power using LSTM and Transformer Model.</li></ul>", unsafe_allow_html=True)

## PROJECTS Section
if section == "PROJECTS":
    #TODO think about this selectbox different type of projects
    project_section = st.selectbox('PROJECTS NAVIGATOR', ["SIDE PROJECTS", "RESEARCH PROJECTS", "ML FROM SCRATCH",
                                                "TRY IT YOURSELF PROJECTS"])
    if project_section == "SIDE PROJECTS":

        #TODO think about the name of this type of projects
        st.header("Side Projects")
        st.subheader("1. Stroke Prediction")
        st.markdown(
            "<ul><li>Built various ML Models such as DecisionTree Classifier and XGBoost Classifier with Undersampling, Oversampling, and SMOTE sampling technique to predict stroke in patient based on gender, age, smoking habit, and other existing conditions.</li>",
            unsafe_allow_html=True)

        with st.expander("Project Report"):
            st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)
            st.write("As per the Centre for Disease Control and Prevention website a stroke, sometimes also known as brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. Which can result into permanent or lasting damage into the brain, sometimes it also causes long-term disability, or even death. According to the World Health Organization stroke is 2nd leading cause of death globally, responsible for around 11% of total deaths.")
            st.write("Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset ")
            st.write("In this project we are going to analyze and predict weather a patient is likely to get a stroke or not based on following input parameters: age, gender, hypertension, heart disease, ever married, work type, residence type, average glucose level, bmi, smoking status.")

            st.markdown("<h3>Data Description</h3>", unsafe_allow_html=True)
            st.write("")
            data_des_col1, data_des_col2 = st.columns([1,1])

            header_html_data_des_df = "<img src='data:image/png;base64,{}' class='img-fluid' width='572.5' height='95' style='display: block;margin-top: 20px'>".format(
                img_to_bytes("images/Projects Content/Stroke Prediction/df.png")
            )
            header_html_data_des_df_info = "<img src='data:image/png;base64,{}' class='img-fluid' width='235' height='195.5' style='display: block;margin-top: 0px'>".format(
                img_to_bytes("images/Projects Content/Stroke Prediction/df_info.png")
            )
            st.markdown(
                header_html_data_des_df, unsafe_allow_html=True,
            )
            st.write("")

            with data_des_col1:
                st.markdown(
                    header_html_data_des_df_info, unsafe_allow_html=True,
                )
            with data_des_col2:
                st.write("Number of records: 5110")
                st.write("Number of features: 12")
                st.write("Categorical features:['gender', 'ever_married', 'work-type', 'Residence_type', 'smoking_status'] ")

            st.markdown("<h3>Data Preparation</h3>", unsafe_allow_html=True)
            st.markdown(
                "<ul><li>After examining the dataset, for feature 'bmi' 201 amount of  null values were found, to handle this null values all the records with feature"
                " 'bmi' where feature 'stroke' has value of 0 was replaced with mean of all the records of feature"
                " 'bmi' with feature 'stroke' value of 0 and the same technique was used for replacing null values for feature 'bmi' and the feature 'stroke' value of 1.</li>"
                "<li>Also it was found that feature 'gender' had only 1 record of value 'other'. This record was dropped since it acts as a Outlier for "
                "this dataset and it won't affect further analysis. </li>"
                "<li>Plus, it was found, dataset contained 5 categorical features, which was converted into int using LabelEncoding.</li>"
                "<li>Plus, it was found, target feature 'stroke' was highly unbalanced, to solve this problem different sampling techniques "
                "like SMOTE, undersampling, oversampling were used. With predefined weights for Machine Learning Algorithm.</li></ul>",
                unsafe_allow_html=True)

            st.markdown("<h3>EDA</h3>", unsafe_allow_html=True)
            st.write("In EDA, relationship between features were examined, where relationship between non-categorical features were examined using "
                     "correlation matrix and categorical features were examined using chi-square test.")
            header_html_eda_corr = "<img src='data:image/png;base64,{}' class='img-fluid' width='358' height='284' style='display: block;margin-top: 10px; margin-left:150px'>".format(
                img_to_bytes("images/Projects Content/Stroke Prediction/corr.png")
            )
            st.markdown(
                header_html_eda_corr, unsafe_allow_html=True,
            )
            st.write("")
            st.write("From the above graph we can infer apart from feature 'bmi' all the other continuous features are somewhat correlated to the 'stroke' where 'bmi' seems to have no linear correlation with 'stroke'.")

            header_html_eda_catcorr = "<img src='data:image/png;base64,{}' class='img-fluid' width='358' height='150' style='display: block;margin-top: 10px; margin-left:150px'>".format(
                img_to_bytes("images/Projects Content/Stroke Prediction/catcorr.png")
            )
            st.markdown(
                header_html_eda_catcorr, unsafe_allow_html=True,
            )
            st.write("")
            st.write("The result of chi-square test stats that there exists a relationship between two variables if p value <=0.5. "
                     "So in our case apart from residence_type all the other features have p value less <= 0.5. "
                     "so there is relationship between gender and stroke, ever_married and stroke, work_type and stroke, smoking status and stroke.")

            st.markdown("<h3>Machine Learning Algorithm Selection</h3>", unsafe_allow_html=True)
            st.write("For this project DecisionTree Classifier and XGBoost Classifier Models were selected. "
                     "Reason behind choosing this models is both of them are tree based classifiers which known to work great with unbalanced dataset. ")

            st.markdown("<h3>Results</h3>", unsafe_allow_html=True)
            st.write("To better understand results first I will define precision and recall")
            st.markdown(
                "<ul><li><b>Precision:</b> Truly predicting class 'stroke (1)'/'non-stroke (0)' upon all the class 'stroke (1)'/'non-stroke (0)' preciditions.</li>"
                "<li><b>Recall:</b> Correctly classifying class 'stroke (1)'/'non-stroke (0)' in case of class 'stroke (1)'/'non-stroke (0)'</li>",
                unsafe_allow_html=True)
            st.write("From above analysis and with model building using Decision tree classifier and XGBClassifier, "
                     "with sampling methods such as SMOTE, Oversampling and Undersampling resulted in following metrics.")
            st.markdown(
                "<ul><li><b>DT with SMOTE:</b> Model overfitted with good training accuracy but moderate test accuracy. Precision and recall for class stroke - poor</li>"
                "<li><b>DT with oversampling:</b> Good training and test accuracy but precision and recall for class 'stoke' - poor</li>"
                "<li><b>DT with undersampling:</b> Model overfitted with poor recall for class 'non-stroke' and poor precision for class 'stroke'</li>"
                "<li><b>XGB without sampling:</b> Perfect Model fitting and accuracy but poor recall for class 'stroke'</li>"
                "<li><b>XGB with SMOTE:</b> Good Model fitting but poor recall and precision for class 'stroke'</li>"
                "<li><b>XGB with oversampling:</b> Poor recall and precision for class 'stroke' with model overfitting</li>"
                "<li><b>XGB with undersampling:</b> Moderate Model fitting and accuracy but poor precision for class 'stroke'</li>",
                unsafe_allow_html=True)

            st.markdown("<h3>Algorithm Selection Reasoning</h3>", unsafe_allow_html=True)
            st.write("After analyzing performance of all the model, "
                     "boiled down to XGB without sampling and XGB with undersampling Model. "
                     "The other model were dicarded because either they were overfitting or they were poorly performing (Precision and Recall). "
                     "The Selection between above two model is the perfect example of trade-off between precision and recall. Lets analyze in depth.")
            st.markdown(
                "<ul><li>For model <b>XGB without sampling</b> we have precision of 1.00 and recall of 0.11 for class 'stoke' meaning when our model predicts 'stroke' is correct 100% whereas it correctly identifies 11% of all cases being 'stroke' when it is 'stroke'.</li>"
                "<li><b>For model <b>XGB with undersampling</b> we have precision of 0.15 and recall of 0.81 for class 'stroke' meaning when our model predicts 'stroke' is correct 15% of the times whereas it correctly identiifies 81% of cases being 'stroke' when it is 'stroke'</li>",
                unsafe_allow_html=True)
            st.write("")
            st.markdown("<p>For our project <b>XGB with undersampling</b> is best since if our model incorretly label 'stroke' does not affect patient that much rather being labeling 'non-stroke' in case of 'stroke'. Plus <b>XGB with undersampling</b> model correctly identiifies 81% of cases being 'stroke' when it is 'stroke'.</p>", unsafe_allow_html=True)
            st.write("See the whole code here")
            st.markdown("""
                               <a href="https://github.com/kpola009/Stroke-Prediction" target = "_blank"> 
                                   <button style="background-color:GreenYellow;">Github</button>
                               </a>
                           """, unsafe_allow_html=True)
            st.write(" ")


        st.subheader("2. Segmenting Miami Areas")
        st.markdown(
            "<ul><li>The goal of this project is to build a K-Means clustering model to group Miami zipcodes based on nearby venue data.</li>",
            unsafe_allow_html=True)
        with st.expander("Project Report"):
            st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)
            st.write("I did this certification on coursera offered by IBM, In which we had to submit a capstone project on segmenting neighborhood of "
                     "toronto city. In this notebook I am performing similar tasks as in capstone project to demonstrate skills which I gained through the course.")
            st.write("In this notebook I will use Foursquare API service to get nearby venues of zipcodes of Miami city. Then using KMeans clustering model to cluster "
                     "zipcodes based on venue's data.")

            st.markdown("<h3>Data Description</h3>", unsafe_allow_html=True)
            st.write("")
            st.write("For this project I have data from three different sources")
            st.markdown(
                "<ul><li>List of Miami ZIPCODES, I got this list from Miami-Dade County website. "
                "<a href=https://gis-mdc.opendata.arcgis.com/datasets/fee863cb3da0417fa8b5aaf6b671f8a7_0/data>Dataset</a></li>"
                "<li>Latitude and Longitude, I got this coordinates using geopy library.</li>"
                "<li>Venue's data, I got this data by calling Foursquare API.</li>",
                unsafe_allow_html=True)
            st.write("")
            st.markdown("<h3>Data Preparation</h3>", unsafe_allow_html=True)
            st.subheader("a. Getting Data")
            header_html_pro_miami = "<img src='data:image/png;base64,{}' class='img-fluid' width='333.33' height='285' style='display: block;margin-top: 10px; margin-left:175px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/Data prep flow.png")
            )
            st.markdown(
                header_html_pro_miami, unsafe_allow_html=True,
            )
            st.write("")

            st.subheader("Prepare Data for the Model")
            st.markdown("<h4>MIAMI ZIPCODE Data</h4>", unsafe_allow_html=True)
            header_html_pro_miami_zio = "<img src='data:image/png;base64,{}' class='img-fluid' width='500' height='185' style='display: block;margin-top: 10px; margin-left:30px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/Zipcode.png")
            )
            st.markdown(
                header_html_pro_miami_zio, unsafe_allow_html=True,
            )
            st.write("")
            st.write("For Miami Zipcode Data, We dropped all the columns aside from column 'ZIPCODE'")
            st.write("Next Step: We fed the collected zipcodes to geopy API to get their latitudes and longitudes. Which resulted in:")
            st.markdown("<h4>LAT & LONG Data</h4>", unsafe_allow_html=True)
            st.write("")
            header_html_pro_miami_long = "<img src='data:image/png;base64,{}' class='img-fluid' width='150' height='208.75' style='display: block;margin-top: 10px; margin-left:30px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/lat_long.png")
            )
            st.markdown(
                header_html_pro_miami_long, unsafe_allow_html=True,
            )
            header_html_pro_miami = "<img src='data:image/png;base64,{}' class='img-fluid' width='606.25' height='208.75' style='display: block;margin-top: 10px; margin-left:30px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/map.png")
            )
            st.markdown(
                header_html_pro_miami, unsafe_allow_html=True,
            )
            st.write("")
            st.write(
                "The above map was created using Folium, it plots ZIPCODES on map using its Latitude and Longitude.")
            st.write("")
            st.write("Next Step: Now we need near by venues information for this zipcodes, for that we feed latitudes and longitudes to "
                     "Foursquare API. Which resulted in:")
            st.markdown("<h4>VENUE Data</h4>", unsafe_allow_html=True)
            header_html_pro_miami_venue = "<img src='data:image/png;base64,{}' class='img-fluid' width='606.25' height='165' style='display: block;margin-top: 10px; margin-left:30px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/venue.png")
            )
            st.markdown(
                header_html_pro_miami_venue, unsafe_allow_html=True,
            )
            st.write("")
            st.write("From the above image we can see, a single zipcode can have multiple values, which is a problem for us since "
                     "we are trying to group zipcodes based on their nearly venues, we need no duplicates for ZIPCODE columns, to handle this "
                     "firstly, 'Venue Category' feature was converted to int using get_dummies function from pandas which resulted in 212 new columns"
                     " and then it was groupedby by 'ZIPCODE' using mean which resulted in:")
            header_html_pro_miami_final = "<img src='data:image/png;base64,{}' class='img-fluid' width='606.25' height='208.75' style='display: block;margin-top: 10px; margin-left:30px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/finaldf.png")
            )
            st.markdown(
                header_html_pro_miami_final, unsafe_allow_html=True,
            )
            st.write("")
            st.markdown("<h3>Building KMeans Model </h3>", unsafe_allow_html=True)
            st.write("To Segment Miami Zipcodes, KMeans model was built with 10 clusters, where number of clusters as a hyperparameter "
                     "was selected using elbow method.")
            header_html_pro_miami_clustered = "<img src='data:image/png;base64,{}' class='img-fluid' width='400' height='420' style='display: block;margin-top: 10px; margin-left:130px'>".format(
                img_to_bytes("images/Projects Content/Segmenting Miami/clusted.png")
            )
            st.markdown(
                header_html_pro_miami_clustered, unsafe_allow_html=True,
            )
            st.write("")
            st.write("Result: As from the map all the zipcodes which are similar based on their nearby venues are grouped into"
                     " same color.")
            st.write("See the whole code here")
            st.markdown("""
                               <a href="https://github.com/kpola009/Miami-Area-Segmenting-with-K-Means" target = "_blank"> 
                                   <button style="background-color:GreenYellow;">Github</button>
                               </a>
                           """, unsafe_allow_html=True)
            st.write("")

        st.subheader("3. Customer Default Prediction")
        st.markdown(
            "<ul><li>The Goal of this project is to predict how likely a customer is going to default, using historical customer data, This project can "
            " be identified as binary classification problem, which was solved using Logistic Regression and Random Forest.</li>",
            unsafe_allow_html=True)
        with st.expander("Project Report"):
            st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)



    # TRY IT YOURSELF PROJECT
    if project_section == "TRY IT YOURSELF PROJECTS":
        st.header("TRY IT YOURSELF PROJECTS")

        st.subheader("1. Pneumonia Classification using Chest X-ray Images")
        st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)
        st.write("The goal of this project is to classify if a person is suffering from pneumonia or not using chest X-Ray images"
                 ". This is a binary classification problem which was solved by training convolution neural network (CNN) and leveraging"
                 " transfer learning.")

        with st.expander("Try it yourself"):
            st.write("Pneumonia is a kind of lung infection, that can cause mild to severe illness. By American Thoracic Society "
                     " report, pneumonia is number 1 reason for US children in hospital. About 1 million adults seek care in for pneumonia "
                     "every year, form which 50,000 people die due to pneumonia.")
            st.write("Based on information from National heart, Lung, and Blood Institute pneumonia is diagnosed based on "
                     "medical history, "
                     " physical exam, and test result. Sometime it is hard to diagnose pneumonia due to pneumonia haveing same  symptoms as"
                     " cold or flu which make problem worth solving.")

            image1, image2, image3 = st.columns([1,1,1])

            header_html_pro_image1 = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/Pneumonia Classification/IM-0187-0001.jpeg")
            )
            header_html_pro_image2 = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/Pneumonia Classification/person1_bacteria_2.jpeg")
            )
            header_html_pro_image3 = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/Pneumonia Classification/person2_bacteria_3.jpeg")
            )

            with image1:
                st.markdown(header_html_pro_image1, unsafe_allow_html=True)
                st.write("IMAGE 1")

            with image2:
                st.markdown(header_html_pro_image2, unsafe_allow_html=True)
                st.write("IMAGE 2")

            with image3:
                st.markdown(header_html_pro_image3, unsafe_allow_html=True)
                st.write("IMAGE 3")

            select_image = st.selectbox('SELECT A IMAGE TO PREDICT', [' ','IMAGE 1', 'IMAGE 2', 'IMAGE 3'])

            if select_image is not None:
                prediction = pneumonia_classification_model.classify_pneumonia(select_image)
                if prediction == 1:
                    st.write("PREDICTION: PNEUMONIA")
                elif prediction == 0:
                    st.write("PREDICTION: NORMAL")

        st.write("See the whole code here")
        st.markdown("""
                           <a href="https://github.com/kpola009/Chest-X-Ray-Pneumonia-Classification-using-Pytorch-and-Tensorflow" target = "_blank"> 
                               <button style="background-color:GreenYellow;">Github</button>
                           </a>
                       """, unsafe_allow_html=True)


        st.subheader("2. GAN (Generative Adversarial Network) (Paper2Code)")
        st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)
        st.write("As the name suggests, GAN are generative models which are built using an adversarial process, in which two models are built"
                 " simultaneously: a generative model G that captures the data distribution, and a discriminative model D that estimates the"
                 " probability that a sample came from the training data rather than G. [Site paper]")
        st.write("Here Model G is trained such that, G generates data which D cannot discriminate whereas model D"
                 " is trained such that, it correctly discriminates generated data by G. Result of this adversarial process"
                 " generates data which follows/similar to training data distribution.")
        with st.expander("Try it yourself"):
            st.caption(
                "Since GAN can be computationally heavy, this might take little time to show the output, Plus we will reduced the dimension of the image to address the computation "
                "time for this project.")
            button_upload, button_exist = st.columns([1,1])

            with button_upload:
               image_button = st.button('Upload Image')

            with button_exist:
                existing_button = st.button('Existing Image')

            if "image_button_state" not in st.session_state:
                st.session_state.image_button_state = False
            if image_button or st.session_state.image_button_state:
                st.session_state.image_button_state = True
                uploaded_image = st.file_uploader("Upload Image", type=['png','jpeg'])
                if uploaded_image is not None:
                    img = load_img(uploaded_image)
                    transform = transforms.Compose([transforms.Resize((128, 128))])
                    img = Image.fromarray(img, "RGB")
                    img = transform(img)

                    st.image(img)
                    epochs = st.slider("Number of epochs", 0, 500)
                    st.write(epochs)
                    if epochs <= 10:
                        st.caption(str(epochs) + " epoch will result in poor image generation.")
                        st.caption("Model started training")

                    if epochs != 0:
                        st.caption("Model started training")
                        im = Image.fromarray(np.uint8(img)).convert('RGB')
                        output = GAN.GAN(im, epochs)
                        output = np.transpose(output.resize(3, 128, 128).cpu().detach().numpy(), (1, 2, 0))
                        output_image = Image.fromarray(output, "RGB")
                        print(output_image)
                        st.image(output_image)

            if existing_button:
                st.write("existing")

        st.write("See the whole code here")
        st.markdown("""
            <a href="https://github.com/kpola009/Paper2Code-GAN" target = "_blank"> 
                <button style="background-color:GreenYellow;">Github</button> 
            </a>
        """, unsafe_allow_html=True)













