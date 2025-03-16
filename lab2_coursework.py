import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler,RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import argparse
from scipy import stats

def compare_results(df,col1,col2):
    statistic, p_value = stats.wilcoxon(df[col1], df[col2])
    print(f'{col1} : {col2} Wilcoxon test statistic: {statistic}, p-value: {p_value}')
    alpha = 0.05
    if p_value < alpha:
        print(f"{col2} result are significantly better")
    else:
        print(f"{col2} result are not better")

def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible
    """
    param_degree = 2
    param_alpha = 1.0
    parser = argparse.ArgumentParser(description="lab2 coursework")
    parser.add_argument("--degree", "-d",type=int, default=2, help="degree for PolynomialFeatures")
    parser.add_argument("--alpha", "-a",type=float, default=1.0, help="alpha for Ridge")
    args = parser.parse_args()

    param_degree=args.degree
    param_alpha = args.alpha

    print(f"degree={param_degree}")
    print(f"alpha={param_alpha}")


    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3  # Modify this value to change the number of repetitions
    train_frac = 0.7  # Modify this value to change the training data fraction (e.g., 0.7 for 70%)
    random_seed = 1 # The random seed will be altered for each repeat

    sn = 1
    results = {'SN': [], 'System': [], 'Dataset': [], 'LR_MAPE': [], 'LR_MAE': [], 'LR_RMSE': [], 'DR_MAPE': [],
               'DR_MAE': [], 'DR_RMSE': []}
    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system) # Modify this to specify the location of the datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')] # List all CSV files in the directory

        # Initialize a dict to store results for average of the metrics
        for csv_file in csv_files:
            print('\n> sn:{}, System: {}, Dataset: {}, Training data fraction: {}, Number of repets: {}'.format(sn, current_system, csv_file, train_frac, num_repeats))
            data = pd.read_csv(os.path.join(datasets_location, csv_file)) # Load data from CSV file
            metrics = {'LR_MAPE': [], 'LR_MAE': [], 'LR_RMSE': [],'DR_MAPE': [], 'DR_MAE': [], 'DR_RMSE': []} # Initialize a dict to store results for repeated evaluations

            for current_repeat in range(num_repeats): # Repeat the process n times
                # Randomly split data into training and testing sets
                colName = "time"
                if current_system=="h2":
                    colName ="throughput"
                #print(X.columns)

                train_data = data.sample(frac=train_frac,
                                          random_state=random_seed * current_repeat)  # Change the random seed based on the current repeat
                test_data = data.drop(train_data.index)
                '''
                #train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat) # Change the random seed based on the current repeat
                #test_data = data.drop(train_data.index)
                '''
                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                '''
                scaler = StandardScaler()
                training_X1 = scaler.fit_transform(training_X0)
                training_X = pd.DataFrame(training_X1, columns=data.columns)
                testing_X = pd.DataFrame(testing_X0)

                print(training_X)
                '''

                #Linear Regression model
                LR_model = LinearRegression() # Initialize a Linear Regression model
                LR_model.fit(training_X, training_Y) # Train the model with the training data
                LR_predictions = LR_model.predict(testing_X) # Predict the testing data
                # Calculate evaluation metrics for the current repeat
                LR_mape = mean_absolute_percentage_error(testing_Y, LR_predictions)
                LR_mae = mean_absolute_error(testing_Y, LR_predictions)
                LR_rmse = np.sqrt(mean_squared_error(testing_Y, LR_predictions))

                scaler = MinMaxScaler()
                training_X1 = scaler.fit_transform(training_X)
                training_X2 = pd.DataFrame(training_X1, columns=data.columns[:-1])
                testing_X1 = scaler.fit_transform(testing_X)
                testing_X2 = pd.DataFrame(testing_X1, columns=data.columns[:-1])
                training_X=training_X2
                testing_X=testing_X2

                poly = PolynomialFeatures(param_degree)  # 选择二次多项式
                X_train_poly = poly.fit_transform(training_X)
                X_test_poly = poly.transform(testing_X)


                #DR_model = Lasso(alpha=0.1)
                #DR_model = Ridge(param_alpha, fit_intercept=True,solver="auto")
                #DR_model.fit(X_train_poly, training_Y)
                #DR_predictions = DR_model.predict(X_test_poly)

                #DR_model = LinearRegression()
                #DR_model.fit(X_train_poly, training_Y)
                #DR_predictions = DR_model.predict(X_test_poly)
                #DR_model = make_pipeline(MinMaxScaler(), HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=5000))
                #DR_model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=5000))
                DR_model = HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=5000)
                DR_model.fit(X_train_poly, training_Y)
                DR_predictions = DR_model.predict(X_test_poly)


                #DecisionTree Regression model
                #DR_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                #DR_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
                #DR_model = GaussianProcessRegressor()
                #DR_model = DecisionTreeRegressor(max_depth=5) # Initialize a DecisionTree  Regression model
                #DR_model.fit(training_X, training_Y) # Train the model with the training data
                #DR_predictions = DR_model.predict(testing_X) # Predict the testing data
                # Calculate evaluation metrics for the current repeat
                DR_mape = mean_absolute_percentage_error(testing_Y, DR_predictions)
                DR_mae = mean_absolute_error(testing_Y, DR_predictions)
                DR_rmse = np.sqrt(mean_squared_error(testing_Y, DR_predictions))

                # Store the metrics
                metrics['LR_MAPE'].append(LR_mape)
                metrics['LR_MAE'].append(LR_mae)
                metrics['LR_RMSE'].append(LR_rmse)
                metrics['DR_MAPE'].append(DR_mape)
                metrics['DR_MAE'].append(DR_mae)
                metrics['DR_RMSE'].append(DR_rmse)

            # Store the result
            results['SN'].append(sn)
            results['System'].append(current_system)
            results['Dataset'].append(csv_file)
            results['LR_MAPE'].append(round(np.mean(metrics['LR_MAPE']), 2))
            results['LR_MAE'].append(round(np.mean(metrics['LR_MAE']), 2))
            results['LR_RMSE'].append(round(np.mean(metrics['LR_RMSE']), 2))
            results['DR_MAPE'].append(round(np.mean(metrics['DR_MAPE']), 2))
            results['DR_MAE'].append(round(np.mean(metrics['DR_MAE']), 2))
            results['DR_RMSE'].append(round(np.mean(metrics['DR_RMSE']), 2))
            sn=sn+1
            # Calculate the average of the metrics for all repeats
            #print('Average MAPE: {:.2f}'.format(np.mean(metrics['LR_MAPE'])))
            #print("Average MAE: {:.2f}".format(np.mean(metrics['LR_MAE'])))
            #print("Average RMSE: {:.2f}".format(np.mean(metrics['LR_RMSE'])))

    df = pd.DataFrame(results)

    count_MAPE = 0
    count_MAE = 0
    count_RMSE = 0

    for row in range(len(results['SN'])):
        if (results['DR_MAPE'][row] < results['LR_MAPE'][row]) : count_MAPE = count_MAPE + 1
        if (results['DR_MAE'][row] < results['LR_MAE'][row]): count_MAE = count_MAE + 1
        if (results['DR_RMSE'][row] < results['LR_RMSE'][row]): count_RMSE = count_RMSE + 1

    avg_LR_MAPE = round(np.mean(results['LR_MAPE']), 2)
    avg_DR_MAPE = round(np.mean(results['DR_MAPE']), 2)
    avg_LR_MAE = round(np.mean(results['LR_MAE']), 2)
    avg_DR_MAE = round(np.mean(results['DR_MAE']), 2)
    avg_LR_RMSE = round(np.mean(results['LR_RMSE']), 2)
    avg_DR_RMSE = round(np.mean(results['DR_RMSE']), 2)

    print(f"Count of DR_MAPE<LR_MAPE is {count_MAPE} , results of {round(count_MAPE/sn*100,2)}% are better")
    print(f"Count of DR_MAPE<LR_MAPE is {count_MAE} , results of {round(count_MAE/sn*100,2)}% are better")
    print(f"Count of DR_MAPE<LR_MAPE is {count_RMSE} , results of {round(count_RMSE/sn*100,2)}% are better")

    print(f"Average LR_MAPE={avg_LR_MAPE}, Average DR_MAPE={avg_DR_MAPE}, avg_DR_MAPE/avg_LR_MAPE = {round(avg_DR_MAPE/avg_LR_MAPE*100)}%")
    print(f"Average LR_MAE={avg_LR_MAE}, Average DR_MAE={avg_DR_MAE} ,avg_DR_MAE/avg_LR_MAE = {round(avg_DR_MAE/avg_LR_MAE*100)}%")
    print(f"Average LR_RMSE={avg_LR_RMSE}, Average DR_RMSE={avg_LR_RMSE}, avg_DR_RMSE/avg_LR_RMSE={round(avg_DR_RMSE/avg_LR_RMSE*100)}%")

    compare_results(df,'LR_MAPE','DR_MAPE')
    compare_results(df, 'LR_MAE', 'DR_MAE')
    compare_results(df, 'LR_RMSE', 'DR_RMSE')

    df.to_csv("lab2.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
