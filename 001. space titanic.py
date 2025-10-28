# spaceship_titanic_analysis.py

import csv
import pandas as pd
import matplotlib.pyplot as plt


class TitanicMerger:
    """train.csv와 test.csv를 합쳐서 data.csv 생성"""

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = 'data.csv'

    def merge_files(self):
        with open(self.output_path, 'w', newline='', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)
            first_file = True

            for file_path in [self.train_path, self.test_path]:
                with open(file_path, 'r', encoding='utf-8') as in_file:
                    reader = csv.reader(in_file)
                    header = next(reader)

                    if first_file:
                        writer.writerow(header)
                        first_file = False

                    for row in reader:
                        writer.writerow(row)

class TitanicAnalyzer:
    """Spaceship Titanic 데이터 분석 및 시각화"""
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, sep=',', encoding='utf-8')

        # Transported Boolean -> 숫자형 변환
        self.df['Transported'] = self.df['Transported'].map({True: 1, False: 0})

    def plot_age_transport(self):
        """나이대별 Transported 비율 막대그래프"""
        self.df['AgeGroup'] = (self.df['Age'] // 10) * 10
        age_transport = self.df.groupby('AgeGroup')['Transported'].mean()
        age_transport.plot(kind='bar', color='skyblue')
        plt.title('Age Group vs Transported')
        plt.xlabel('Age Group')
        plt.ylabel('Transported Ratio')
        plt.tight_layout()
        plt.show()

    def plot_destination_age_distribution(self):
        """Destination별 나이대 분포 시각화"""
        self.df['AgeGroup'] = (self.df['Age'] // 10) * 10
        destinations = self.df['Destination'].dropna().unique()

        for dest in destinations:
            subset = self.df[self.df['Destination'] == dest]
            age_counts = subset['AgeGroup'].value_counts().sort_index()
            age_counts.plot(kind='bar', alpha=0.6)
            plt.title(f'Age Distribution for Destination {dest}')
            plt.xlabel('Age Group')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    # 1. CSV 파일 병합
    merger = TitanicMerger('train.csv', 'test.csv')
    merger.merge_files()

    # 2. 데이터 분석 및 시각화
    analyzer = TitanicAnalyzer('data.csv')
    analyzer.plot_age_transport()
    analyzer.plot_destination_age_distribution()
