import pandas as pd


def main():
    df = pd.read_csv('2022254004_안성인_iaq_data.csv')
    df.to_pickle('2022254004_siahn.pkl')

if __name__ == "__main__":
    # print __doc__
    main()
