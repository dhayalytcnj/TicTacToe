
'''
Yash Dhayal, Michael Giordano, Corbin Grosso, & Yuriy
CSC 426-01

Project 1: Self-Learning Tic Tac Toe
'''

import numpy
# import matplotlib.pyplot as plt #not on cluster, used to get graphs

#global variables for win, loss, and tie counts, as well as total nnumber of games played
WIN_COUNT = 0
LOSS_COUNT = 0
TIE_COUNT = 0
TOTAL_GAMES = 0

#initializes lists to store win lose and tie rate.
winRate = []
loseRate = []
tieRate = []
totalTracker = []   #Why do we need this? 

class GameState:
    #initializing some variables
    def __init__(self, p1, p2):
        self.board = numpy.zeros((3, 3))  #3x3 board with just zeroes
        self.p1 = p1
        self.p2 = p2
        self.gameEnd = False    # check if game is over
        self.boardHash = None
        self.whoseTurn = 1      # init p1 plays first
        self.features = [0 for i in range(3)]      # init to a list of three zeroes; tracks the boards current features
        self.V_train = 0    # init to 0; will be calculated before use


    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(3 * 3))
        return self.boardHash


    # switching player turns after a position is filled
    def updateState(self, position):
        self.board[position] = self.whoseTurn
        if self.whoseTurn == 1:
            self.whoseTurn = -1 
        else:
            self.whoseTurn = 1
        self.calculateFeaturesValues()
        self.calculateV_train(numpy.array(list(self.board)), self.whoseTurn)
        self.p1.updateWeights(self.features, self.V_train)


    # determines the value of the board features
    def calculateFeaturesValues(self):
        self.features[2] = self.board[1,1] # features[2] tracks who, if anyone, has taken the center space
        for i in range(3):
            # features[0] is the total number of rows, columns, and diagonals that have 2 of Player 1's spaces and 1 open space
            # features[1] is the total number of rows, columns, and diagonals that have 2 of Player 2's spaces and 1 open space
            temp = [self.board[i,0], self.board[i,1], self.board[i,2]] # checks each column
            if sum(temp) == 2: # if the sum is 2, then there has to be two 1s and a 0. No other combination could make this
                self.features[0] += 1
            if sum(temp) == -2: # if the sum is -2, then there has to be two -1s and a 0. No other combination could make this
                self.features[1] += 1
            temp = [self.board[0,i], self.board[1,i], self.board[2,i]] # checks each row
            if sum(temp) == 2:
                self.features[0] += 1
            if sum(temp) == -2:
                self.features[1] += 1
        temp = [self.board[0,0], self.board[1,1], self.board[2,2]] # checks upper left diagonal
        if sum(temp) == 2:
            self.features[0] += 1
        if sum(temp) == -2:
            self.features[1] += 1
        temp = [self.board[2,0], self.board[1,1], self.board[0,2]] # checks upper right diagonal
        if sum(temp) == 2:
            self.features[0] += 1
        if sum(temp) == -2:
            self.features[1] += 1

    
    # Calculate V_train recursively
    def calculateV_train(self, current_board, current_turn):
        positions = self.availableSpots()
        action = self.p1.chooseAction(positions, current_board, current_turn)
        # take action and upate board state
        current_board[action] = current_turn
        # check board status if it is end
        win = self.winnerPredict()
        if win is None:
            self.calculateV_train(numpy.array(list(current_board)), -current_turn)
        else:
            return win



    # reset board
    def reset(self):
        self.board = numpy.zeros((3, 3))
        self.boardHash = None
        self.gameEnd = False
        self.whoseTurn = 1


    #determining empty/not claimed positions on board
    def availableSpots(self):
        openSpots = []
        for x in range(3):
            for y in range(3):
                if self.board[x, y] == 0:
                    openSpots.append((x, y))  # need to be tuple
        return openSpots


    #determining winner
    def winner(self):
        # Checking if sum of rows = 3 (for p1 to win) or -3 (for p2 to win)
        for x in range(3):
            if sum(self.board[x, :]) == 3:
                self.gameEnd = True
                return 1
            if sum(self.board[x, :]) == -3:
                self.gameEnd = True
                return -1
        
        # Checking if sum of columns = 3 (for p1 to win) or -3 (for p2 to win)
        for x in range(3):
            if sum(self.board[:, x]) == 3:
                self.gameEnd = True
                return 1
            if sum(self.board[:, x]) == -3:
                self.gameEnd = True
                return -1

        # Checking if sum of diagonals = 3 (for p1 to win) or -3 (for p2 to win)
            #OLD CODE: diag_sum1 = sum([self.board[x, x] for x in range(3)]) #bottom left to top right diagonal
            #OLD CODE: diag_sum2 = sum([self.board[x, 3 - x - 1] for x in range(3)]) #top left to bottom right diagonal
        diag_sum1 = 0
        diag_sum2 = 0
        for x in range(3):  
            diag_sum1 += self.board[x, x]           #bottom left to top right diagonal sum
            diag_sum2 += self.board[x, 3 - x - 1]   #top left to bottom right diagonal
        
        #Finding which diagonal sum is higher. Abs used for player 2, since their values are negative
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.gameEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1    #p1 wins
            else:
                return -1   #p2 wins

        #Checking for tie if no possible positions left and since no prior win conditions were met
        if len(self.availableSpots()) == 0:
            self.gameEnd = True
            return 0
        
        #if none of the prior conditions were met, then the game isn't over
        self.gameEnd = False
        return None


    #determining winner
    def winnerPredict(self):
        # Checking if sum of rows = 3 (for p1 to win) or -3 (for p2 to win)
        for x in range(3):
            if sum(self.board[x, :]) == 3:
                return 1
            if sum(self.board[x, :]) == -3:
                return -1
        
        # Checking if sum of columns = 3 (for p1 to win) or -3 (for p2 to win)
        for x in range(3):
            if sum(self.board[:, x]) == 3:
                return 1
            if sum(self.board[:, x]) == -3:
                return -1

        # Checking if sum of diagonals = 3 (for p1 to win) or -3 (for p2 to win)
            #OLD CODE: diag_sum1 = sum([self.board[x, x] for x in range(3)]) #bottom left to top right diagonal
            #OLD CODE: diag_sum2 = sum([self.board[x, 3 - x - 1] for x in range(3)]) #top left to bottom right diagonal
        diag_sum1 = 0
        diag_sum2 = 0
        for x in range(3):  
            diag_sum1 += self.board[x, x]           #bottom left to top right diagonal sum
            diag_sum2 += self.board[x, 2 - x]   #top left to bottom right diagonal
        
        #Finding which diagonal sum is higher. Abs used for player 2, since their values are negative
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1    #p1 wins
            else:
                return -1   #p2 wins

        #Checking for tie if no possible positions left and since no prior win conditions were met
        if len(self.availableSpots()) == 0:
            return 0
        
        #if none of the prior conditions were met, then the game isn't over
        return None


    # only when game ends
    def winPoints(self):
        result = self.winner()
        # assigning win points
        global WIN_COUNT, LOSS_COUNT, TIE_COUNT, TOTAL_GAMES
        if result == 1:     #p1 wins
            self.p1.setWinPoints(1)
            self.p2.setWinPoints(0)
            WIN_COUNT += 1
        elif result == -1:  #p2 wins
            self.p1.setWinPoints(0)
            self.p2.setWinPoints(1)
            LOSS_COUNT += 1
        else:   #tie
            self.p1.setWinPoints(0)
            self.p2.setWinPoints(0)
            TIE_COUNT += 1
        TOTAL_GAMES += 1
        
        #used to keep track of outcomes for plots
        winRate.append(WIN_COUNT / TOTAL_GAMES * 100)
        loseRate.append(LOSS_COUNT / TOTAL_GAMES * 100)
        tieRate.append(TIE_COUNT / TOTAL_GAMES * 100)
        totalTracker.append(TOTAL_GAMES)


    # How a match will occur
    def play(self, rounds):
        for i in range(rounds):
            if i % 1000 == 0:
                print(f"Rounds simulated: {i}")
            while not self.gameEnd:
                # AI 1 - See chooseAction method in AI class
                positions = self.availableSpots()
                p1_action = self.p1.chooseAction(positions, self.board, self.whoseTurn)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:  # game ended with p1 either win or draw
                    self.winPoints()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # AI 2 - CHANGE THIS. Must be fixed alg
                    positions = self.availableSpots()
                    p2_action = self.p2.chooseAction(positions, self.board, self.whoseTurn)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # ended with p2 either win or draw
                        self.winPoints()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break


class AI:
    def __init__(self, name, learning_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
<<<<<<< HEAD
        self.learning_rate = learning_rate  # determine rate at which the AI learns
        self.lr = 0.2   #THIS NEEDS TO GO
        self.exp_rate = exp_rate    #THIS NEEDS TO GO
        self.decay_gamma = 0.9      #THIS NEEDS TO GO
>>>>>>> 4234b382dd236816f4f81221e59ea7e7a265b4f7
        self.states_value = {}  # state -> value
        self.weights = [0 for i in range(3)] # initialize to a list of 3 zeroes


    # reset board
    def reset(self):
        self.states = []


    def getHash(self, board):
        boardHash = str(board.reshape(3 * 3))
        return boardHash


    # append a hash state
    def addState(self, state):
        self.states.append(state)

    

    # Calculate a new value for each of the weights
    def updateWeights(self, features, V_train):
        V_hat = 0
        for i in range(len(self.weights)):
            V_hat += self.weights[i] * features[i] # Calculate V_hat = w1x1 + w2x2 + w3x3

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learning_rate * (V_train - V_hat) * features[i]


    # determine AI action
    def chooseAction(self, positions, current_board, symbol):
        if numpy.random.uniform(0, 1) <= self.learning_rate:
            # take random action
            action = positions[numpy.random.choice(len(positions))]
        else:
            value_max = -999
            for x in positions:
                next_board = current_board.copy()
                next_board[x] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = x
        return action


    # at the end of game, backpropagate and update states value
    def setWinPoints(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

'''
    def savePolicy(self):
            fw = open('save.txt', 'w+')
            .dump(self.states_value, fw)
            fw.close()
'''

if __name__ == "__main__":
    # training
    p1 = AI("p1")
    p2 = AI("p2")

    st = GameState(p1, p2)
    training = input("How many matches do you want the AI to self-train: ")
    while not training.isnumeric(): # error checking user input
        print("That is not a number. Try again.\n")
        training = input("How many matches do you want the AI to self-train: ")

    # Displaying outputs
    print("Training in process")
    st.play(int(training))
    print("TRAINING COMPLETE")
    print("Wins: ", WIN_COUNT)
    print("Losses: ", LOSS_COUNT)
    print ("Ties: ", TIE_COUNT)

    #prints out the graphs of winRate, loseRate, and tieRate. Only works if (matplotlib.pyplot) is installed
    # plt.plot(totalTracker, winRate)
    # plt.title('Win Tracker')
    # plt.xlabel('Number of Games Played')
    # plt.ylabel('Win Percentage')
    # plt.ylim([-1, 101])
    # plt.xlim([1, TOTAL_GAMES])
    # plt.show()

    # plt.plot(totalTracker, loseRate)
    # plt.title('Lose Tracker')
    # plt.xlabel('Number of Games Played')
    # plt.ylabel('Lose Percentage')
    # plt.ylim([-1, 101])
    # plt.xlim([1, TOTAL_GAMES])
    # plt.show()

    # plt.plot(totalTracker, tieRate)
    # plt.title('Tie Tracker')
    # plt.xlabel('Number of Games Played')
    # plt.ylabel('Draw Percentage')
    # plt.ylim([-1, 101])
    # plt.xlim([1, TOTAL_GAMES])
    # plt.show()
