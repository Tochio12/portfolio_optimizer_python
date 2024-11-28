from flask import Flask, render_template, request, redirect, url_for # Web framework to build a web application
import yfinance as yf # Libary to fetch stock data from Yahoo Finance
import pandas as pd # Used for handling data tables like an excel sheet
from datetime import datetime, timedelta # Toold for working wiht dates and times
import numpy as np # For numerical operations and calculations
from scipy.optimize import minimize # Used to optimize investment strategy 
import matplotlib.pyplot as plt # Used for creating plots and graphs
from fredapi import Fred # Fetches economic data from the Federal Reserve ex: interest rates
import os # Working with file paths

# Create a flask web application instance 
app = Flask(__name__)

# Initialized the Fred API for fetching economic data
fred = Fred(api_key='cd4acb5e62326da28b29d47daab6a595')

# Define the main route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    # When the user submits the form post request 
    if request.method == "POST":
        # Get the tickers entered by the user 
        tickers_input = request.form.get("tickers")
        # Clean up the input split by commas, remove exrtra spaces and convert to uppercase
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        
        # Check that exactly 5 stock tickers were provided
        if len(tickers) != 5:
            return "Please enter exactly 5 stock tickers, separated by commas."
        
        try:
            # Define the time range for fetching historical stock data
            end_date = datetime.today() # Today's date
            start_date = end_date - timedelta(days = 10 * 365) # 10 years ago

            # DataFrame to store adjusted closing prices for all stocks 
            adj_close_df = pd.DataFrame()

            # Fetch historical stock data for each ticker
            for ticker in tickers:
                # Download stock data using Yahoo Finance
                data = yf.download(ticker, start = start_date, end = end_date)
                # Add the "Adj Close" prices to the DataFrame 
                adj_close_df[ticker] = data["Adj Close"]
            
            # Calculate daily log returns for each stock
            # Log returns = ln(current price / previous price) used to measure percentage changes in price
            log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

            # Calculate the covariance matrix of log returns how much the stock move relative to each other
            # Multiply by 252 to annualize since there are 252 trading days in a year
            cov_matrix = log_returns.cov() * 252
            
            # This function calculates the portfolio's standard deviation or risk
            def standard_deviation(weights, cov_matrix):
                # Variance = weights transpose * covariance matrix * weights
                variance = weights.T @ cov_matrix @ weights
                # Standard deviation = square root of variance
                return np.sqrt(variance)
            
            # Function to calculate the portfolio's expected return 
            def expected_return(weights, log_returns):
                # Multiply average daily return by weights and annualize 252 trading days
                return np.sum(log_returns.mean() * weights) * 252
            
            # Function to calculate the Sharpe Ratio risk adjusted return 
            def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
                # Sharpe Ratio = (Expected Return - Risk-Free Rate) / Risk (standard deviation)
                return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
            
            # Function to minimize negative Sharpe Ratio since we want to maxmize Sharpe Ratio
            def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
                return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
            
            # Get the current 10 year Treasury rate proxy for risk free rate
            ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
            risk_free_rate = ten_year_treasury_rate.iloc[-1] # Use the latest rate 
            
            # Constraits for optimization weights must sum to 1 all money is invested
            constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            # Bounds each stock can have between 0% and 50% of the portfolio
            bounds = [(0, 0.5) for _ in range(len(tickers))]
            # Initial guess divide money equally among the 5 stocks 
            initial_weights = np.array([1 / len(tickers)] * len(tickers))
            
            # Use the minimize function to optimize the portfolio
            optimized_results = minimize(
                neg_sharpe_ratio, # Minimize negative Sharpe Ratio
                initial_weights, # Start with equal weight
                args=(log_returns, cov_matrix, risk_free_rate), # arguments for the function
                method='SLSQP', # Sequential Least Squares Programming
                constraints=constraints, # Ensure weight sum to 1
                bounds=bounds # Ensure weight stay within 0-50%
            )

            # Extract the optimal weights from the optimization results
            optimal_weights = optimized_results.x
            
            
            # Save the plot as an image file
            plot_path = os.path.join("static", "optimal_weights.png")  # Save it to the static folder
            plt.figure(figsize=(10, 6)) # create a figure

            # Create a bar chart of optimal weights 
            plt.bar(tickers, optimal_weights * 100)  # Convert to percentage for display
            plt.xlabel('Assets') # label for x-axis 
            plt.ylabel('Optimal Weights (%)')  # Label for y-axis
            plt.title('Optimal Portfolio Weights') # Chart title 

            # Save the chart as a image 
            plt.savefig(plot_path)
            plt.close()  # Close the plot to free memory 

            # Combine tickers and weights into a list of tuples
            ticker_weights = list(zip(tickers, optimal_weights))  # Zip the tickers and weights together

            # Pass the image file path and ticker_weights to the result.html template
            return render_template("result.html", ticker_weights=ticker_weights, plot_url=plot_path)
        except Exception as e:
            # Handle any errors that occur and display the error message 
            return f"An error occurred: {str(e)}"
    
    # If it's a get request render the index.html template
    return render_template("index.html")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)