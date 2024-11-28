from flask import Flask, render_template, request, redirect, url_for # Web framework to build a web application
import yfinance as yf # Libary to fetch stock data from Yahoo Finance
import pandas as pd # Used for handling data tables like an excel sheet
from datetime import datetime, timedelta # Toold for working wiht dates and times
import numpy as np # For numerical operations and calculations
from scipy.optimize import minimize # Used to optimize investment strategy 
import matplotlib.pyplot as plt # Used for creating plots and graphs
from fredapi import Fred # Fetches economic data from the Federal Reserve ex: interest rates
import os

app = Flask(__name__)

# Initialize FRED API
fred = Fred(api_key='cd4acb5e62326da28b29d47daab6a595')

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tickers_input = request.form.get("tickers")
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        
        if len(tickers) != 5:
            return "Please enter exactly 5 stock tickers, separated by commas."
        
        try:
            # Portfolio optimization logic
            end_date = datetime.today()
            start_date = end_date - timedelta(days=10 * 365)
            adj_close_df = pd.DataFrame()
            for ticker in tickers:
                data = yf.download(ticker, start=start_date, end=end_date)
                adj_close_df[ticker] = data["Adj Close"]
            
            log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
            cov_matrix = log_returns.cov() * 252
            
            def standard_deviation(weights, cov_matrix):
                variance = weights.T @ cov_matrix @ weights
                return np.sqrt(variance)
            
            def expected_return(weights, log_returns):
                return np.sum(log_returns.mean() * weights) * 252
            
            def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
                return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
            
            def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
                return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
            
            # Get risk-free rate
            ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
            risk_free_rate = ten_year_treasury_rate.iloc[-1]
            
            # Optimization setup
            constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            bounds = [(0, 0.5) for _ in range(len(tickers))]
            initial_weights = np.array([1 / len(tickers)] * len(tickers))
            
            # Optimization
            optimized_results = minimize(
                neg_sharpe_ratio,
                initial_weights,
                args=(log_returns, cov_matrix, risk_free_rate),
                method='SLSQP',
                constraints=constraints,
                bounds=bounds
            )
            optimal_weights = optimized_results.x
            
            
            # return render_template("result.html", tickers=tickers, weights=optimal_weights, image_data=image_data)
            # Save the plot as an image file
            plot_path = os.path.join("static", "optimal_weights.png")  # Save it to the static folder
            plt.figure(figsize=(10, 6))

            # Update to display percentages on the chart
            plt.bar(tickers, optimal_weights * 100)  # Convert to percentage for display
            plt.xlabel('Assets')
            plt.ylabel('Optimal Weights (%)')  # Label in percentage
            plt.title('Optimal Portfolio Weights')

            plt.savefig(plot_path)
            plt.close()  # Close the plot to free resources

            # Pass the image file path to the result.html template
            # Combine tickers and weights into a list of tuples
            ticker_weights = list(zip(tickers, optimal_weights))  # Zip the tickers and weights together

            # Pass the image file path and ticker_weights to the result.html template
            return render_template("result.html", ticker_weights=ticker_weights, plot_url=plot_path)
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
