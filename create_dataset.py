import pandas as pd

'''

def compress_basic_data(filename):
    """
    A function to choose all the films with the value "movie"
    """
    basic_data = open(filename, 'r', encoding="utf8").readlines()
    movie_basic_data = open("title_movies_basics.tsv", 'w', encoding='utf8')
    movie_basic_data.write(basic_data[0])
    for line in basic_data:
        data = line.split('\t')
        if data[1] == 'movie':
            movie_basic_data.write(line)
    movie_basic_data.close()

data = open('Data/Box_office_directors.csv', encoding='utf8').readlines()
print(len(data))
print("reading basics data...")
basics_data = pd.read_table('Data/title_movies_basics.tsv')
# Removing all the movies without a start year
basics_data = basics_data[basics_data.startYear != '\\N']
# Converting all startYears to ints
basics_data['startYear'] = basics_data['startYear'].astype('int')
# Sorting the startYears with the most recent one first
basics_data = basics_data.sort_values(by='startYear', ascending=False)
# Picking the movies made between 1990 and 2018
basics_data = basics_data[basics_data.startYear >= 1990]
basics_data = basics_data[basics_data.startYear <= 2018]
# Changing the index from numbers to tconst
basics_data.index = basics_data.tconst

print("reading crew data...")
crew_data = pd.read_table('Data/title_crew.tsv')

print("reading ratings data...")
ratings_data = pd.read_table('Data/title_ratings.tsv')

# Set indexes to tconst
crew_data.index = crew_data.tconst
ratings_data.index = ratings_data.tconst

# Labels from each file to be included in the new file
basics_labels = ['primaryTitle', 'genres']
crew_labels = ['directors', 'writers']
rating_labels = ['averageRating']

# Concatenates the columns and keeps only the columns
#movies_dataset = pd.concat([basics_data[basics_labels], crew_data[crew_labels], ratings_data[rating_labels]],
#                        axis=1, join='inner')

#movies_dataset.to_csv('movies_dataset.csv')
'''

# Open movies dataset
movies_dataset = pd.read_csv('movies_dataset.csv')
movies_dataset.index = movies_dataset.primaryTitle
print(movies_dataset.head())

# Open box office dataset
bo_dataset = pd.read_csv('number_title_boxoffice.csv', sep=';')
bo_dataset.columns = ['no', 'primaryTitle', 'gross']
bo_dataset.index = bo_dataset.primaryTitle


# Concatenate files
movies_labels = ['genres', 'directors', 'writers', 'averageRating']
bo_labels = ['gross']
bo_movies_dataset = pd.merge(movies_dataset[movies_labels], bo_dataset[bo_labels], left_index=True, right_index=True, how='outer')
print(bo_movies_dataset.head())
bo_movies_dataset = bo_movies_dataset.dropna()

bo_movies_dataset.to_csv('Box_office_movies_dataset.csv')
