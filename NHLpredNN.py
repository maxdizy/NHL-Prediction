import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(0)

team1 = 'Toronto'
gameData1 = {'gameId':  [0],
        'gameDate': [20220125],
        'home_or_away': [1],
        'avgGoalDif': [2],
        'avgHF': [17.125],
        'winStreak': [3],
        'avgPPG': [4.5],
        'daysSinceLast': [1],
        }

game1Df = pd.DataFrame(gameData1)

team2 = 'Vancouver'
gameData2 = {'gameId':  [0],
        'gameDate': [20220125],
        'home_or_away': [0],
        'avgGoalDif': [5],
        'avgHF': [14.25],
        'winStreak': [0],
        'avgPPG': [3.4],
        'daysSinceLast': [1],
        }

game2Df = pd.DataFrame(gameData2)

def cleanData(df):
    df = df.loc[(df['situation'] == 'all')]
    #only take what we want
    df = df[['team','gameId','season','opposingTeam','gameDate','home_or_away','xGoalsFor','xGoalsAgainst','goalsFor','goalsAgainst','hitsFor','hitsAgainst']]
    #determine a win
    df['win'] = df['goalsFor'] > df['goalsAgainst']
    #sort the df by team and then by date
    df = df.sort_values(['team','gameDate']).reset_index(drop=True)

    #how many previous games to calculate the avg from
    avgSize = 8

    df[['team','gameId','season','gameDate','goalsFor','goalsAgainst','hitsFor','hitsAgainst','win']].head(20)
    
    #create new avrg metrics
    metrics = ['daysSinceLast','points','winStreak','avgGF','avgGA','avgxGF','avgxGA','avgHF','avgHA']
    avgMetrics = [('goalsFor','avgGF'),('goalsAgainst','avgGA'),('xGoalsFor','avgxGF'),('xGoalsAgainst','avgxGA'),('hitsFor','avgHF'),('hitsAgainst','avgHA')]

    for metric in metrics: 
        df[metric] = np.zeros(len(df))

    #create game numbers
    df.loc[0,'game'] = 1
    for i in range(0,len(df)-1):
        if (df.loc[i,'gameDate'] > df.loc[i+1,'gameDate'] or df.loc[i,'season'] != df.loc[i+1,'season']):
            #reset the start of each season game number to 1
            df.loc[i+1,'game'] = 1
        else:
            #game number (for simplicity purposes)
            df.loc[i+1,'game'] = 1+ df.loc[i,'game']

    # fill the new metrics based on previous game data
    for i in range(0,len(df)-1):
        if (df.loc[i,'game'] > df.loc[i+1,'game']):
            #reset each stat to 0 at the start of each team's data
            for metric in metrics:
                df.loc[i+1,metric] = 0
        else:
            #days since last game
            lastGame = pd.to_datetime(df.loc[i,'gameDate'], format='%Y%m%d')
            thisGame = pd.to_datetime(df.loc[i+1,'gameDate'], format='%Y%m%d')
            df.loc[i+1,'daysSinceLast'] = (thisGame - lastGame).days
            #points
            df.loc[i+1,'points'] += (2*df.loc[i,'win'] + df.loc[i,'points'])
            #winstreak
            if df.loc[i,'win']:
                df.loc[i+1,'winStreak'] = df.loc[i,'winStreak'] + 1
            else:
                df.loc[i+1,'winStreak'] = 0
                
            #compute all the average metrics using the list
            #avgMetrics = ['avgGF','avgGA','avgxGF','avgxGA','avgHF','avgHA']
            for metric,avgMetric in avgMetrics:
                value = 0
                #only look at games after avgSize games (i.e. if avgSize = 10, then only look at games after 10)
                if df.loc[i+1,'game'] > avgSize:
                    start,end = i+1 - avgSize, i+1
                    for gameIndex in range(start,end):
                        value += df.loc[gameIndex,metric]
                    df.loc[i+1,avgMetric] = value / avgSize
                else:
                    df.loc[i+1,avgMetric] = np.nan

    #avg point columns
    df['avgPPG'] = np.zeros(len(df))

    for i in range(0,len(df)-1):
        if (df.loc[i,'game'] > df.loc[i+1,'game']):
            #reset avgPPG to 0 at the start of each team's data
            df.loc[i+1,'avgPPG'] = 0
        else:
            totalPointsInSet = 0
            if df.loc[i+1,'game'] > avgSize: 
                start,end = i+1 - avgSize,i+1
                for gameIndex in range(start,end):
                    totalPointsInSet += 2*df.loc[gameIndex,'win']
                df.loc[i+1,'avgPPG'] = totalPointsInSet / avgSize
            else:
                df.loc[i+1,avgMetric] = np.nan   

    differentials = [('avgGoalDif','avgGF','avgGA'),('avgXGoalDif','avgxGF','avgxGA')]
    for dif,For,Against in differentials:
        df[dif] = df[For] - df[Against]

    #change home to 1 and away to 0
    df["home_or_away"] = df['home_or_away'].replace('HOME', 1) 
    df["home_or_away"] = df['home_or_away'].replace('AWAY', 0) 

    #fix missing values
    df["avgGoalDif"].fillna(0, inplace=True)
    df["avgXGoalDif"].fillna(0, inplace=True)
    df["avgHF"].fillna(0, inplace=True)
    
    matchups = df[['gameId','gameDate','home_or_away','avgGoalDif','avgHF','winStreak','avgPPG','daysSinceLast', 'win']].set_index(['gameId'])

    matchups.to_csv('cleanedData.csv')
    return pd.read_csv('cleanedData.csv')

def result(gamePred, team):
    if gamePred > 0.5:
        return print(team + " will win the game with %.2f%% certainty" %(gamePred*100))
    if gamePred < 0.5:
        return print(team + " will lose the game with %.2f%% certainty" % ((1-gamePred)*100))
    return print("The game will be tied")

og1 = pd.read_csv('TOR.csv')
df1 = cleanData(og1)

og2 = pd.read_csv('VAN.csv')
df2 = cleanData(og2)

#seperate team 1 data into x and y
y1 = df1['win']
x1 = df1.drop('win', axis=1)

#split data into testing and training data
x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.2, random_state=100) #will split 20% of data into test data

#seperate team 2 data into x and y
y2 = df2['win']
x2 = df2.drop('win', axis=1)

#split data into testing and training data
x2Train, x2Test, y2Train, y2Test = train_test_split(x2, y2, test_size=0.2, random_state=100) #will split 20% of data into test data

#build neural net model
mlp = MLPRegressor(hidden_layer_sizes=(150,100,50), max_iter=300 ,activation='relu', solver='adam')
# mlp = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(100), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#        random_state=None, shuffle=True, solver='adam', tol=0.0001,
#        validation_fraction=0.1, verbose=False, warm_start=False)

mlp1 = mlp.fit(x1Train, y1Train)
mlp2 = mlp.fit(x2Train, y2Train)

#apply model to make a prediction for team 1
predictTrain1 = mlp1.predict(x1Train)
predictTest1 = mlp1.predict(x1Test)

#apply model to make a prediction for team 1
predictTrain2 = mlp2.predict(x2Train)
predictTest2 = mlp2.predict(x2Test)

#data visualization of prediction results for team 1
plt.scatter(x=y1Train, y=predictTrain1, c='#7CAE00', alpha=0.3)
plt.ylabel('Predicted Win Team 1')
plt.xlabel('Experimental Win Team 1')
plt.show()

#data visualization of prediction results for team 2
plt.scatter(x=y2Train, y=predictTrain2, c='#7CAE00', alpha=0.3)
plt.ylabel('Predicted Win Team 2')
plt.xlabel('Experimental Win Team 2')
plt.show()

scaler = MinMaxScaler()

gamePred1 = mlp1.predict(game1Df)
gamePred2 = mlp2.predict(game2Df)
print(predictTest1)
result(gamePred1, team1)
result(gamePred2, team2)