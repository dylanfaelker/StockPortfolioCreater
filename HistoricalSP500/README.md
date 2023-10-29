This folder is independant of the rest of the repo

monthly_csv_to_history_table
- a jupyter file to convert monthly bloomberg SP500 consituent data into a csv
- the monthly data needs to be stored in ./data folder

SP500_monthly_hist - master.csv
- 10 years of monthly consituent data

SP500_monthly_hist.csv
- constituent data used in the simulation

multi_year_simulation.ipynb
- file to generate the back dated profolio
- to run:
    - save SP500_monthly_hist.csv in the same folder with the dates you want to calculate for
    - run all cells of the file in order
    - result is saved in portfolio_vs_sp500_hist.json
