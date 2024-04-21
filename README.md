# Food Delivery Time Prediction using ML
<img src="assets/food_delivery_README.jpg" width="400">

## Overview

This project aims to implement a predictive model for accurately estimately the food delivery time using ML. The repository is structured with several key components for data analysis, modelling and deployment of the predictive model.

**About Dataset**

<font size="2">Food delivery is a courier service in which a restaurant, store, or independent food-delivery company delivers food to a customer. An order is typically made either through a restaurant or grocer's website or mobile app, or through a food ordering company. The delivered items can include entrees, sides, drinks, desserts, or grocery items and are typically delivered in boxes or bags. The delivery person will normally drive a car, but in bigger cities where homes and restaurants are closer together, they may use bikes or motorized scooters. Prompt and accurate delivery time directly impacts customer satisfaction and influences their overall experience.</font>

## Project objective

<font size="2">The project aims to develop an accurate real-time food delivery time prediction system by considering crucial factors such as distance, historical delivery data, and various influencing variables. By calculating the geographical separation between the food preparation point and delivery location, we establish a foundational parameter for estimating delivery duration. Historical data on delivery times for similar distances is analyzed to identify patterns and dependencies, allowing the creation of a predictive model.

This predictive model leverages machine learning algorithms and statistical techniques to incorporate not only distance but also factors like traffic conditions, time of day, day of the week, weather conditions, and delivery partner workload.

Regular updates with the latest delivery data ensure the model remains relevant and adaptable to changing conditions, making it a dynamic and responsive system. In real-time, as new orders are received and delivery partners are assigned, the model recalculates and adjusts estimated delivery times, providing customers with up-to-date and reliable information.</font>

## Data-dictionary

|Column|Description |
| :------------ |:---------------:|
|**ID**|order ID number| 
|**Delivery_person_ID**|ID number of the delivery partner|
|**Delivery_person_Age**|Age of the delivery partner|
|**Delivery_person_Ratings**|Ratings of the delivery partner based on past deliveries|
|**Restaurant_latitude**|The latitude of the restaurant|
|**Restaurant_longitude**|The longitude of the restaurant|
|**Delivery_location_latitude**|The latitude of the delivery location|
|**Delivery_location_longitude**|The longitude of the delivery location|
|**Order_Date**|Date of the order|
|**Time_Orderd**|Time the order was placed|
|**Time_Order_picked**|Time the order was picked|
|**Weatherconditions**|Weather conditions of the day|
|**Road_traffic_density**|Density of the traffic|
|**Vehicle_condition**|Condition of the vehicle|
|**Type_of_order**|The type of meal ordered by the customer|
|**Type_of_vehicle**|The type of vehicle delivery partner rides|
|**multiple_deliveries**|Amount of deliveries driver picked|
|**Festival**|If there was a Festival or no.|
|**City_type**|Type of city, example metropolitan, semi-urban, urban.|
|**Time_taken(min)**| The time taken by the delivery partner to complete the order|

## Decision-making process

The primary goal of the project was to develop an effective prediction model for estimating order delivery times in minutes, utilizing a combination of diverse features as predictors. This endeavor involved applying various transformation techniques, conducting feature engineering, and selecting features to uncover concealed patterns and correlations among them.

By exploiting these discovered relationships, the objective was to construct a resilient prediction model capable of accurately estimating delivery times for individual orders. The project aimed to transcend mere data analysis and delve into the underlying factors influencing delivery times, thereby enabling more precise predictions and improving overall operational efficiency.

The entire process of tackling the problem and progressing towards a solution was thoroughly documented in a series of notebooks. Each notebook delineated a distinct stage of the project, outlining the systematic approach employed to analyze, transform, engineer, and select features for the prediction model.

**Notebooks:**

1. [problem_statement](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/00_problem_statement.ipynb)

2. [complete_pipeline](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/05_complete_pipeline.ipynb)

3. [data_cleaning](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/01_data_cleaning.ipynb)

4. [data_eda](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/02_data_eda.ipynb)

5. [feature_engineering](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/03_feature_engineering.ipynb)

6. [preprocessing_modelling_feature_selection](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/04_preprocessing_modelling_feature_selection.ipynb)

These notebooks acted as thorough documentation of the methodologies utilized, detailing any preprocessing techniques applied to the data, transformation methods employed for features, and strategies implemented for feature engineering.

Moreover, the notebooks recorded the precise algorithms, models, or techniques employed for prediction. This documentation not only promoted transparency and reproducibility but also served as a valuable asset for future reference, enabling effortless sharing of insights, methodologies, and discoveries with peers and stakeholders. By systematically documenting the entire project within these notebooks, it ensured a comprehensive comprehension of the process and established a basis for future enhancements and iterations.


## Setting Up the Project

### Prerequisites

- Docker installed on your machine.

### Instructions

1. Clone the repository:

    ```bash
    git clone  https://github.com/PrepVector/applied-ml-uber-eta-prediction.git
    ```

2. Build the Docker image:

    ```bash
    docker build -t food_delivery_time .
    ```

3. Run the Docker container:

    ```bash
    docker run -p 8501:8501 food_delivery_time
    ```

4. Access the Streamlit app in your web browser at [http://localhost:8501](http://localhost:8501).

### Additional Commands

- To enter the Docker container shell:

    ```bash
    docker run -it food_delivery_time /bin/bash
    ```

- To stop the running container:

    ```bash
    docker stop $(docker ps -q --filter ancestor=food_delivery_time)
    ```

Adjust the instructions based on your specific project needs.
