**Interest Rate Modeling: CIR & CIR++ Calibration**
This project is a Python tool that predicts and models interest rates. It compares two classic financial models: the Cox-Ingersoll-Ross (CIR) model and its upgraded version, CIR++ - to see how accurately they track real-world market changes.
The tool tests these models against historical data to evaluate their accuracy and automatically applies an error-correction feature to fix prediction mistakes.

**How the Code Works**
The program is broken down into 5 simple steps:

**Data Prep & Filtering**: Cleans up interest rate data, fills in any missing values, and gives extra weight to the most recent market trends.
**Base CIR Calibration**: Runs a math engine that calculates the best baseline settings (like rate volatility and adjustment speed) to match current market conditions.
**Yield Curve Building**: Uses the current short-term interest rate to predict what longer-term interest rates will look like down the road.
**CIR++ Upgrade**: Adds an error-correction layer. By learning from its past mistakes, the model adjusts its predictions to better match real-world market shapes, especially for longer-term rates.
**Diagnostics**: Checks prediction accuracy for each individual timeframe and flags moments where standard financial formulas break down due to extreme market volatility.

**Requirements** 
Technical Dependencies
The project runs on standard Python data science libraries:
pandas and numpy (for loading and organizing data)
scipy (for finding the best mathematical settings)
scikit-learn (for measuring prediction accuracy)
matplotlib and seaborn (for drawing the final performance charts)

**Running the Project**
To run the model, download this repository to your computer and open your terminal or command prompt. Navigate to the project's folder, ensure your Python environment has the required libraries installed, and launch the primary script.

