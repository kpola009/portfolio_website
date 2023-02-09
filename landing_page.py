import streamlit as st
import numpy as np
import streamlit.components.v1 as components
from PIL import Image
import base64
from pathlib import Path
import pneumonia_classification_model
from time import sleep

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

    st.markdown("<h1>KETUL POLARA</h1>", unsafe_allow_html=True)

    pages = ["ABOUT ME", "EDUCATION/CERTIFICATIONS", "EXPERIENCE", "PROJECTS"]
    section = st.sidebar.radio('', pages)

if section == 'ABOUT ME':
    st.markdown("<h2>ABOUT ME</h2>", unsafe_allow_html=True)
    picture, content = st.columns([1, 1])
    with picture:
        header_html_my_picture = "<img src='data:image/png;base64,{}' class='img-fluid' width='234.25' height='280.5' style='display: block;margin-top: 10px'>".format(
            img_to_bytes("images/1.jpg")
        )
        st.markdown(
            header_html_my_picture, unsafe_allow_html=True,
        )
        st.markdown("<h4>KETUL POLARA</h4>", unsafe_allow_html=True)
        # st.markdown("<h4>KETUL POLARA</h4>", unsafe_allow_html=True)
        st.write("Email: kpola009@fiu.edu")
    with content:
        st.write("As a recent graduate with a background in Information Technology and a strong interest in Data Science. I have gained experience with programming, conducting research, designing, building, and deploying Machine Learning models, with Data Visualization, Data Analysis, and Statistics through working as Machine Learning Researcher at Energy Systems Research Laboratory (FIU) and from coursework. I possess the ability to uncover valuable insights, make well-informed decisions, and tackle complex business problems using statistical techniques and cutting-edge data-driven approaches.")


    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.markdown("""
                   <a href="https://github.com/kpola009" target = "_blank"> 
                       <button style="background-color:GreenYellow;">Github</button> 
                   </a>
               """, unsafe_allow_html=True)

    with col1:
        st.markdown("""
                           <a href="https://github.com/kpola009/portfolio_website/blob/master/Resume%20Ketul%20Polara%20.pdf" target = "_blank"> 
                               <button style="background-color:GreenYellow;">Resume</button> 
                           </a>
                       """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
                           <a href="https://linkedin.com/in/ketul-polara" target = "_blank"> 
                               <button style="background-color:GreenYellow;">LinkedIn</button> 
                           </a>
                       """, unsafe_allow_html=True)


    st.write("")
    st.write("")

    st.markdown("<h3>Technical Skills & Tools</h3>", unsafe_allow_html=True)

    col_pro1, col_pro2 = st.columns([2,3])

    with col_pro1:
        st.write("Programming Languages: ")
        st.write("")
        st.write("Machine Learning: ")
        st.write("")
        st.write("Deep Learning: ")
        st.write("")
        st.write("Data Visualization: ")
        st.write("")
        st.write("Others:")
        st.write("")
        st.write("Exposure: ")
        st.write("")
        st.write("Cloud: ")
    with col_pro2:
        st.button("Python | SQL | Java")
        st.button("Scikit-Learn | Pandas | Numpy")
        st.button("Tensorflow | Keras | PyTorch")
        st.button("Matplotlib | Plotly | Seaborn")
        st.button("Streamlit | Git | Excel")
        st.button("Flask | Hadoop | Spark")
        st.button("AWS (S3, EC2, SageMaker) | Azure (Data Lake, Data Factory)")

if section == 'EDUCATION/CERTIFICATIONS':
    ### EDUCATION Section
    st.markdown("<h2>EDUCATION AND CERTIFICATIONS</h2>", unsafe_allow_html=True)
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

    st.markdown("<h2>CERTIFICATIONS</h2>", unsafe_allow_html=True)
    st.markdown(
        "<ul><li><b>IBM Data Science Professional</b>, IBM</li>"
        "<li><b>Advanced Data Science Specialization</b>, IBM</li>"
        "<li><b>Big Data Specialization</b>, University of San Diego</li>"
        "<li><b>Deep Learning Specialization</b>, deeplearning.ai</li>"
        "<li><b>Azure AI Fundamentals</b>, Microsoft</li></ul>",
        unsafe_allow_html=True)



### EXPERIENCE Section
if section == "EXPERIENCE":
    st.markdown("<h2>EXPERIENCE</h2>", unsafe_allow_html=True)
    st.write("")

    experience_col1, experience_col2 = st.columns([1,2])

    header_html_exp = "<img src='data:image/png;base64,{}' class='img-fluid' width='160'  style='display: block;margin-top: 200px'>".format(
        img_to_bytes("images/Energy_Systems_Lab_Logo_Final-e1472763643607.png")
    )
    header_html_exp_1 = "<img src='data:image/png;base64,{}' class='img-fluid' width='160'  style='display: block;margin-top: 250px'>".format(
        img_to_bytes("images/1577947253254.png")
    )
    with experience_col1:

        st.markdown(
            header_html_exp, unsafe_allow_html=True,
        )

        st.markdown(
            header_html_exp_1, unsafe_allow_html=True,
        )

    with experience_col2:
        st.markdown("<h2>Machine Learning Researcher (GRA)</h2>", unsafe_allow_html=True)
        st.markdown("<h4>Energy Systems Research Laboratory (FIU)</h4>", unsafe_allow_html=True)
        st.write("August 2021 - Current")
        st.markdown("<ul><li>Designed and developed Time-series database (InfluxDB) to capture data from 200 data points using RTI Data distributions service (Similar to Apache Kafka) for smart grid testbed.</li>"
                    "<li>Using AWS SageMaker, S3, and SageMaker endpoint, trained and deployed Autoencoder and Isolation Forest Machine Learning models, in python for anomaly detection in smart grid testbed.</li>"
                    "<li>Implemented Federated learning Framework Flower with ANN and CNN to detect DoS attacks on IoT devices.</li>"
                    "<li>Implemented Forecasting models for Load and Solar Power using LSTM and Transformer Model.</li></ul>", unsafe_allow_html=True)

        st.markdown("<h2>Data Engineer, Intern </h2>", unsafe_allow_html=True)
        st.markdown("<h4>Apexx Strategies</h4>", unsafe_allow_html=True)
        st.write("Mar 2020 - Dec 2020")
        st.markdown(
            "<ul><li>Performed Data Collection (SQL), Data Cleaning (Python), and Data Visualization.</li>"
            "<li>Develop deep understanding of the data sources, implement data standards, and maintain data quality.</li>"
            "<li>Developed a pipeline to perform full loading of data from OLTP source to Azure Data Lake in CSV format using Azure Data Factory.</li></ul>",
            unsafe_allow_html=True)

## PROJECTS Section
if section == "PROJECTS":
    #TODO think about this selectbox different type of projects
    project_section = st.selectbox('PROJECTS NAVIGATOR', [" ","SIDE PROJECTS", "TRY IT YOURSELF PROJECTS", "RESEARCH PROJECTS", "ML FROM SCRATCH"])

    # b1, b2,b3,b4 = st.columns([1,1,1,1])
    # with b1:
    #     b1_click = st.button("SIDE PROJECTS")
    # with b2:
    #     b2_click = st.button("TRY IT YOURSELF PROJECTS")
    # with b3:
    #     b3_click = st.button("RESEARCH PROJECTS")
    # with b4:
    #     b4_click =st.button("ML FROM SCRATCH")


    if project_section == "SIDE PROJECTS":

        #TODO think about the name of this type of projects
        st.header("Side Projects")

        st.subheader("1. Stroke Prediction")
        stroke_prediction_thumbnail = "<img src='data:image/png;base64,{}' class='img-fluid' width='600' height='300' style='display: block;margin-top:0px'>".format(
                img_to_bytes("images/Projects Content/Thumbnails/stroke.png"))
        st.markdown(
            stroke_prediction_thumbnail, unsafe_allow_html=True,
        )
        st.write("")
        st.write("Built various ML Models such as DecisionTree Classifier and XGBoost Classifier with Undersampling, Oversampling, and SMOTE sampling technique to predict stroke in patient based on gender, age, smoking habit, and other existing conditions.")

        with st.expander("Project Report"):
            st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)
            st.write("As per the Centre for Disease Control and Prevention website a stroke, sometimes also known as brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. Which can result into permanent or lasting damage into the brain, sometimes it also causes long-term disability, or even death. According to the World Health Organization stroke is 2nd leading cause of death globally, responsible for around 11% of total deaths.")
            st.write("Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset ")
            st.write("In this project we are going to analyze and predict weather a patient is likely to get a stroke or not based on following input parameters: age, gender, hypertension, heart disease, ever married, work type, residence type, average glucose level, bmi, smoking status. The goal is to identify potential risk factors and provide a risk assessment for the individual, which can be used by healthcare professionals to take preventative measures and reduce the risk of stroke. It is important to note that the prediction should not be considered as a definite diagnosis and that a comprehensive medical evaluation by a healthcare professional is necessary to determine the presence of stroke or any other medical conditions.")

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
                "<ul><li>After analyzing the dataset, it was found that 201 records for the 'bmi' feature had null values. To handle these null values, the mean of all 'bmi' values was calculated where the 'stroke' feature had a value of 0. This mean value was then used to replace the null values in the corresponding records for the 'bmi' feature and 'stroke' value of 0. The same approach was used for replacing the null values of the 'bmi' feature where the 'stroke' value was 1.</li>"
                "<li>Additionally, it was discovered that the 'gender' feature had only one record with a value of 'other.' This record was removed as it acted as an outlier in the dataset and would not have any significant impact on further analysis. </li>"
                "<li>Furthermore, the dataset contained 5 categorical features which were transformed into numerical values using the Label Encoding technique.</li>"
                "<li>Additionally, it was observed that the target feature 'stroke' was highly imbalanced. To address this issue, various sampling techniques such as SMOTE, undersampling, and oversampling were applied with predefined weights for the machine learning algorithm.</li></ul>",
                unsafe_allow_html=True)

            st.markdown("<h3>EDA</h3>", unsafe_allow_html=True)
            st.write("In the Exploratory Data Analysis (EDA), the relationships between the features were analyzed. The correlation between the non-categorical features was examined using a correlation matrix,"
                     " while the relationships between the categorical features were analyzed using a chi-square test.")
            header_html_eda_corr = "<img src='data:image/png;base64,{}' class='img-fluid' width='358' height='284' style='display: block;margin-top: 10px; margin-left:150px'>".format(
                img_to_bytes("images/Projects Content/Stroke Prediction/corr.png")
            )
            st.markdown(
                header_html_eda_corr, unsafe_allow_html=True,
            )
            st.write("")
            st.write("From the above graph, we can observe that all the continuous features (excluding 'bmi') appear to have some degree of correlation with the 'stroke' outcome. However, 'bmi' does not seem to have a linear correlation with 'stroke'.")

            header_html_eda_catcorr = "<img src='data:image/png;base64,{}' class='img-fluid' width='358' height='150' style='display: block;margin-top: 10px; margin-left:150px'>".format(
                img_to_bytes("images/Projects Content/Stroke Prediction/catcorr.png")
            )
            st.markdown(
                header_html_eda_catcorr, unsafe_allow_html=True,
            )
            st.write("")
            st.write("The results of the chi-square test indicate that there is a relationship between two variables if the p-value "
                     " is less than or equal to 0.5. In this case, all features (excluding residence type) have a p-value "
                     " less than or equal to 0.5, indicating a relationship between the feature and 'stroke'. "
                     " This includes a relationship between 'gender' and 'stroke', 'ever_married' and 'stroke', 'work_type' and 'stroke', and 'smoking status' and 'stroke'.")

            st.markdown("<h3>Machine Learning Algorithm Selection</h3>", unsafe_allow_html=True)
            st.write("For this project, the Decision Tree Classifier and XGBoost Classifier models were selected. The reason for choosing these models is that they are both tree-based classifiers, which have been known to perform well on imbalanced datasets.")

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
            st.write("After evaluating the performance of all the models, "
                     " it was determined that the best models were XGBoost without sampling and XGBoost with undersampling. "
                     " The other models were discarded because they either overfitted the data or had poor precision and recall scores. "
                     " The choice between these two models illustrates the trade-off between precision and recall. "
                     " Let's examine this in more detail.")
            st.markdown(
                "<ul><li>For model <b>XGB without sampling</b> we have precision of 1.00 and recall of 0.11 for class 'stoke' meaning when our model predicts 'stroke' is correct 100% whereas it correctly identifies 11% of all cases being 'stroke' when it is 'stroke'.</li>"
                "<li>For model <b>XGB with undersampling</b> we have precision of 0.15 and recall of 0.81 for class 'stroke' meaning when our model predicts 'stroke' is correct 15% of the times whereas it correctly identiifies 81% of cases being 'stroke' when it is 'stroke'</li>",
                unsafe_allow_html=True)
            st.write("")
            st.markdown("<p>For this project, the <b>XGBoost model with undersampling</b> is considered to be the best model. This is because incorrect labeling of a 'non-stroke' case does not have a significant impact on the patient, compared to incorrect labeling of a 'stroke' case as 'non-stroke.' Additionally, the XGBoost model with undersampling correctly identifies 81% of cases as 'stroke' when it is actually 'stroke.'</p>", unsafe_allow_html=True)

        st.markdown("""
                           <a href="https://github.com/kpola009/Stroke-Prediction" target = "_blank"> 
                               <button style="background-color:GreenYellow;">See the whole code here</button>
                           </a>
                       """, unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")

        st.subheader("2. Segmenting Miami Areas")
        stroke_prediction_thumbnail = "<img src='data:image/png;base64,{}' class='img-fluid' width='600' height='300' style='display: block;margin-top:0px'>".format(
            img_to_bytes("images/Projects Content/Thumbnails/miami.png"))
        st.markdown(
            stroke_prediction_thumbnail, unsafe_allow_html=True,
        )
        st.write("")
        st.write("The goal of this project is to build a K-Means clustering model to group Miami zipcodes based on nearby venue data.")

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

            st.subheader("b. Prepare Data for the Model")
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

        st.markdown("""
                           <a href="https://github.com/kpola009/Miami-Area-Segmenting-with-K-Means" target = "_blank"> 
                               <button style="background-color:GreenYellow;">See the whole code here</button>
                           </a>
                       """, unsafe_allow_html=True)
        st.write("")
        st.write("")

        st.subheader("3. Customer Default Prediction")
        stroke_prediction_thumbnail = "<img src='data:image/png;base64,{}' class='img-fluid' width='600' height='300' style='display: block;margin-top:0px'>".format(
            img_to_bytes("images/Projects Content/Thumbnails/default.png"))
        st.markdown(
            stroke_prediction_thumbnail, unsafe_allow_html=True,
        )
        st.write("")
        st.write("The Goal of this project is to predict how likely a customer is going to default, using historical customer data, This project can "
            " be identified as binary classification problem, which was solved using Logistic Regression and Random Forest.")

        with st.expander("Project Report"):
            st.markdown("<h3>Introduction</h3>", unsafe_allow_html=True)
            st.write("Customer loan defaults refer to the failure of a borrower to repay a loan according to the agreed-upon terms. "
                     " This can happen for a variety of reasons such as a change in financial circumstances. "
                     " Loan defaults can have significant consequences for both the borrower and the lender. "
                     " Understanding the causes of loan defaults and developing strategies to minimize them is a crucial aspect of responsible lending and credit risk management.")
            st.write("In this project as mentioned, machine learning techniques were leveraged to predict the likelihood of customer defaulting. "
                     " By training this Machine learning models on historical data, lender can identify patterns and trends that are associated with higher risk default such as specific "
                     " demographic characteristics or financial behaviors.")

            st.markdown("<h3>Data Description</h3>", unsafe_allow_html=True)
            st.write("")
            data_des_col1, data_des_col2 = st.columns([1, 1])

            header_html_data_des_df_cp = "<img src='data:image/png;base64,{}' class='img-fluid' width='650' height='170' style='display: block;margin-top: 20px'>".format(
                img_to_bytes("images/Projects Content/Customer Default/df.png")
            )
            header_html_data_des_df_info_cp = "<img src='data:image/png;base64,{}' class='img-fluid' width='124.165' height='256.1667' style='display: block;margin-left: 80px'>".format(
                img_to_bytes("images/Projects Content/Customer Default/df_info.png")
            )
            st.markdown(
                header_html_data_des_df_cp, unsafe_allow_html=True,
            )
            st.write("")

            with data_des_col1:
                st.markdown(
                    header_html_data_des_df_info_cp, unsafe_allow_html=True,
                )
            with data_des_col2:
                st.write("Number of records: 94000")
                st.write("Number of features: 32")
                st.write("All the features in the dataset have numeric datatype and anonymized.")

            # st.write("")
            st.markdown("<h3>EDA</h3>", unsafe_allow_html=True)

            st.write("- Initially, we started by analyzing the relationships between features in our dataset using a correlation matrix, "
                " since all of our features are numerical in nature. "
                " A correlation matrix is a useful tool in data analysis as it provides a graphical representation of the relationships "
                " between the numerical features. By computing the correlation coefficients between each pair of features, "
                " the matrix enables us to identify features that are highly correlated with one another, giving us valuable "
                " insights into the structure of the data. These insights can aid in feature selection and help us better understand the "
                " relationships between the features.")

            st.write("")
            header_html_data_corr = "<img src='data:image/png;base64,{}' class='img-fluid' width='446.625' height='487.375' style='display: block;margin-left:35px'>".format(
                img_to_bytes("images/Projects Content/Customer Default/corr.png")
            )
            st.markdown(
                header_html_data_corr, unsafe_allow_html=True,
            )
            st.write("")
            st.write("From the above corr matrix it was found, feature A22 and A24, A6 and A5, A10 and A9, A12 and A8 are highly correlated, by eliminating those features can result in better performing model.")

            st.write("- Secondly, histograms and probability plots were employed to investigate the normality of all the features in the dataset. The histograms provide a visual representation of the distribution of each feature, while the probability plots allow for a more precise assessment of normality by comparing the observed feature values to a theoretical normal distribution. These tools help us determine if the features in the dataset are normally distributed, which can be important in certain statistical tests and modeling techniques that assume normality. By using both histograms and probability plots, we can have a thorough understanding of the normality of the features in the dataset.")
            st.write("From the result of normality check it was found most of the features in dataset were highly skewed.")

            st.write("- Thirdly, boxplots were utilized to check for the presence of outliers in all the features. A boxplot provides a graphical representation of the distribution of a feature by plotting the median, quartiles, and any outliers.")
            st.write("From the boxplots, it was found most of the features contain outliers.")

            header_html_data_im = "<img src='data:image/png;base64,{}' class='img-fluid' width='392.67' height='288' style='display: block;margin-left: 35px'>".format(
                img_to_bytes("images/Projects Content/Customer Default/download.png")
            )
            st.markdown(
                header_html_data_im, unsafe_allow_html=True,
            )
            st.write("")
            st.write("Finally, using histogram it was determined that the target variable in our dataset is imbalanced. Specifically, there are more instances of customers who did not default compared to those who did default.")

            st.markdown("<h3>Data Transformation</h3>", unsafe_allow_html=True)
            st.write("")
            st.markdown(
                "<ul><li>Upon initial examination, it was found that there were no missing values in the dataset. However, a duplicate row was detected and was subsequently removed to maintain the accuracy of the data.</li>"
                "<li>Subsequently, it was observed that all features in the dataset were heavily skewed and contained outliers. To address this issue, the Yeo-Johnson transformation was applied to correct the skewness and handle the outliers.</li>"
                "<li>Thirdly, to handle imbalanced target variable, we will use StratifiedKFold cross validation technique during model training. </li>",
                unsafe_allow_html=True)
            st.write("")
            header_html_data_aft = "<img src='data:image/png;base64,{}' class='img-fluid' width='512' height='281.75' style='display: block;margin-left: 35px'>".format(
                img_to_bytes("images/Projects Content/Customer Default/Drawing7 (1).png")
            )
            st.markdown(
                header_html_data_aft, unsafe_allow_html=True,
            )
            st.write("")
            st.write("Issues which were found during EDA were handled in this section as shown in above plots.")

            st.markdown("<h3>ML Model Building</h3>", unsafe_allow_html=True)
            st.write("Given the nature of the problem, it was deemed a binary classification problem. To make predictions on the probability of customer default, "
                     "both Logistic Regression and Random Forest models were utilized.")

            st.markdown("<h4>a. Logistic Regression</h4>", unsafe_allow_html=True)
            st.write("The sklearn function gridSearchCV was utilized to find the optimal value for the hyperparameter C. It was determined that a value of 0.01 gave the best AUC score. Additionally, to address the imbalanced nature of the target variable, StratifiedKFolds was employed, resulting in an AUC score of 0.844515.")
            st.markdown("<h4>b. Random Forest</h4>", unsafe_allow_html=True)
            st.write("The choice to use Random Forest was made because it is a tree-based algorithm, and it has a reputation for being capable of handling imbalanced data. Upon training the Random Forest algorithm, an AUC score of 0.81 was obtained.")

        st.markdown("""
                           <a href="https://github.com/kpola009/Customer-Default-Prediction" target = "_blank">
                               <button style="background-color:GreenYellow;">See the whole code here</button>
                           </a>
                       """, unsafe_allow_html=True)


    # TRY IT YOURSELF PROJECT
    if project_section == 'TRY IT YOURSELF PROJECTS':
        st.header("TRY IT YOURSELF PROJECTS")

        st.subheader("1. Pneumonia Classification using Chest X-ray Images")
        stroke_prediction_thumbnail = "<img src='data:image/png;base64,{}' class='img-fluid' width='600' height='300' style='display: block;margin-top:0px'>".format(
            img_to_bytes("images/Projects Content/Thumbnails/pne.jpg"))
        st.markdown(
            stroke_prediction_thumbnail, unsafe_allow_html=True,
        )

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
                    st.write(select_image + " PREDICTION: PNEUMONIA")
                elif prediction == 0:
                    st.write(select_image + " PREDICTION: NORMAL")

        st.markdown("""
                           <a href="https://github.com/kpola009/Chest-X-Ray-Pneumonia-Classification-using-Pytorch-and-Tensorflow" target = "_blank"> 
                               <button style="background-color:GreenYellow;">See the whole code here</button>
                           </a>
                       """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")

        st.subheader("2. GAN (Generative Adversarial Network) (Paper2Code)")
        stroke_prediction_thumbnail = "<img src='data:image/png;base64,{}' class='img-fluid' width='600' height='300' style='display: block;margin-top:0px'>".format(
            img_to_bytes("images/Projects Content/Thumbnails/gan.jpg"))
        st.markdown(
            stroke_prediction_thumbnail, unsafe_allow_html=True,
        )

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

            image1_gan, image2_gan, image3_gan = st.columns([1, 1, 1])

            header_html_pro_image1_gan = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/GAN/truck.jpeg")
            )
            header_html_pro_image2_gan = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/GAN/DSC_0365-1000x543.jpg")
            )
            header_html_pro_image3_gan = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/GAN/iss041e067595.jpg")
            )

            with image1_gan:
                st.markdown(header_html_pro_image1_gan, unsafe_allow_html=True)
                st.write("IMAGE 1")

            with image2_gan:
                st.markdown(header_html_pro_image2_gan, unsafe_allow_html=True)
                st.write("IMAGE 2")

            with image3_gan:
                st.markdown(header_html_pro_image3_gan, unsafe_allow_html=True)
                st.write("IMAGE 3")

            select_image_gan = st.selectbox('SELECT A IMAGE TO PREDICT', [' ', ' IMAGE 1', ' IMAGE 2', ' IMAGE 3'])

            header_html_pro_image2_gan_result = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/GAN/download (1).png")
            )
            header_html_pro_image3_gan_result = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/GAN/download (2).png")
            )
            header_html_pro_image1_gan_result = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='200' style='display: block;margin-top: 10px;'>".format(
                img_to_bytes("images/Projects Content/GAN/download.png")
            )

            sleep(5)
            if select_image_gan is not None:
                if select_image_gan == ' IMAGE 1':
                    st.markdown(header_html_pro_image1_gan_result, unsafe_allow_html=True)
                    st.write("GENERATED IMAGE")

                if select_image_gan == ' IMAGE 2':
                    st.markdown(header_html_pro_image2_gan_result, unsafe_allow_html=True)
                    st.write("GENERATED IMAGE")

                if select_image_gan == ' IMAGE 3':
                    st.markdown(header_html_pro_image3_gan_result, unsafe_allow_html=True)
                    st.write("GENERATED IMAGE")

        st.markdown("""
                   <a href="https://github.com/kpola009/Paper2Code-GAN" target = "_blank">
                       <button style="background-color:GreenYellow;">See the whole code here</button>
                   </a>
               """, unsafe_allow_html=True)

    if project_section == 'RESEARCH PROJECTS':
        st.write("COMING SOON")
    if project_section == "ML FROM SCRATCH":
        st.write("COMING SOON")