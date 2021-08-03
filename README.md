# Car-Performance-Prediction
This project is aimed at predicting car's mileage (mpg) using a dataset which contains multiple features like car name, horsepower, number of cylinders and many more. This project uses car-performance dataset, all the preprocessing steps are applied on the data, to prepare the data to train the model.

Multiple machine learning regression algorithms are trained and tested for r_2 score metric. Random Forest turned out to give the most accurate model for predicting of mpg values.

Then the notebook and the model was deployed on IBM cloud platform and was integrated with python flask network, which was connected to the end point of webpage, the webpage takes the values for attributes of a car and predicts it's mpg (miles per gallon) value and the car performance (Low/Moderate/High).

To run the webpage, enter the following command on command prompt (run the python file app.py)- 

python app.py

And then copy the localhost link and paste it on any browser to view the webpage.


# Sample EDA graph

<img width="891" alt="Screenshot 2021-08-03 at 8 45 12 PM" src="https://user-images.githubusercontent.com/73014208/128040709-bbd002e8-e11e-494d-a209-ac044e488bd2.png">


# R_2 Score Comparision

<img width="471" alt="Screenshot 2021-08-03 at 8 46 44 PM" src="https://user-images.githubusercontent.com/73014208/128040962-1fc576a8-82be-4c97-b642-11453a6b08ea.png">


# Sample tree from the final Forest

<img width="858" alt="Screenshot 2021-08-03 at 8 46 09 PM" src="https://user-images.githubusercontent.com/73014208/128040883-86b54380-ac82-4f3c-97f7-3a63719c7042.png">

# Scatter Plot of Predicted and Original mpg values

<img width="855" alt="Screenshot 2021-08-03 at 8 47 27 PM" src="https://user-images.githubusercontent.com/73014208/128041064-d0030493-b5b5-40d6-b850-e18a3ac8d866.png">


