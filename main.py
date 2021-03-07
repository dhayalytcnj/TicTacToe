
'''
Yash Dhayal, Michael Giordano, Corbin Grosso, & Yuriy Deyneka
CSC 426-01

Project 1: Self-Learning Tic Tac Toe
'''

import numpy
import matplotlib.pyplot as plt #not on cluster, used to get graphs

#global variables for win, loss, and tie counts, as well as total nnumber of games played
WIN_COUNT = 0
LOSS_COUNT = 0
TIE_COUNT = 0
TOTAL_GAMES = 0

#initializes lists to store win lose and tie rate.
winRate = []
loseRate = []
tieRate = []
totalTracker = []

class GameState:
    """
    Class containing data about the 
    """
    #initializing some variables
    def __init__(self, p1, p2):
        """
        :param p1: AI player 1, the AI that is being actively trained
        :param p2: AI player 2, the opponent of the AI that is being actively trained
        :type p1: AI class object
        :type p2: AI class object
        """
        self.board = numpy.zeros((3, 3))  #3x3 board with just zeroes
        self.p1 = p1
        self.p2 = p2
        self.gameEnd = False    # check if game is over
        # self.boardHash = None
        self.whoseTurn = 1      # init p1 plays first
        self.features = [0 for i in range(3)]      # init to a list of three zeroes; tracks the boards current features
        self.V_train = 0    # init to 0; will be calculated before use


    # switching player turns after a position is filled
    def updateState(self, position):
        self.board[position] = self.whoseTurn
        if self.whoseTurn == 1:
            self.whoseTurn = -1 
        else:
            self.whoseTurn = 1
        self.calculateFeaturesValues()
        self.calculateV_train(self.board.copy(), self.whoseTurn)
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
        if current_turn == 1:
            action = self.p1.chooseAction(positions, current_board, current_turn)
        else:
            action = self.p2.chooseAction(positions, current_board, current_turn)
        # take action and upate board state
        current_board[action] = current_turn
        # check board status if it is end
        win = self.winnerPredict(current_board)
        if win is None:
            # current_board is typecasted to a list and back to a numpy array in order to avoid altering the original board on accident
            self.calculateV_train(current_board.copy(), -current_turn)
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


    #determining empty/not claimed positions on board
    def availableSpotsPredict(self, current_board):
        openSpots = []
        for x in range(3):
            for y in range(3):
                if current_board[x, y] == 0:
                    openSpots.append((x, y))  # need to be tuple
        return openSpots


    #determining winner
    def winner(self):
        # Checking if sum of rows = 3 (for p1 to win) or -3 (for p2 to win)
        for i in range(3):
            temp = [self.board[i,0], self.board[i,1], self.board[i,2]]
            if sum(temp) == 3:
                self.gameEnd = True
                return 1
            elif sum(temp) == -3:
                self.gameEnd = True
                return -1
        
        # Checking if sum of columns = 3 (for p1 to win) or -3 (for p2 to win)
        for i in range(3):
            temp = [self.board[0,i], self.board[1,i], self.board[2,i]]
            if sum(temp) == 3:
                self.gameEnd = True
                return 1
            elif sum(temp) == -3:
                self.gameEnd = True
                return -1

        # Checking if sum of diagonals = 3 (for p1 to win) or -3 (for p2 to win)
        temp = [self.board[0,0], self.board[1,1], self.board[2,2]]
        if sum(temp) == 3:
            self.gameEnd = True
            return 1
        elif sum(temp) == -3:
            self.gameEnd = True
            return -1

        temp = [self.board[0,2], self.board[1,1], self.board[2,0]]
        if sum(temp) == 3:
            self.gameEnd = True
            return 1
        elif sum(temp) == -3:
            self.gameEnd = True
            return -1

        #Checking for tie if no possible positions left and since no prior win conditions were met
        if len(self.availableSpots()) == 0:
            self.gameEnd = True
            return 0
        
        #if none of the prior conditions were met, then the game isn't over
        self.gameEnd = False
        return None


    #determining winner
    def winnerPredict(self, current_board):
        # Checking if sum of rows = 3 (for p1 to win) or -3 (for p2 to win)
        for i in range(3):
            temp = [current_board[i,0], current_board[i,1], current_board[i,2]]
            if sum(temp) == 3:
                return 1
            elif sum(temp) == -3:
                return -1
        
        # Checking if sum of columns = 3 (for p1 to win) or -3 (for p2 to win)
        for i in range(3):
            temp = [current_board[0,i], current_board[1,i], current_board[2,i]]
            if sum(temp) == 3:
                return 1
            elif sum(temp) == -3:
                return -1

        # Checking if sum of diagonals = 3 (for p1 to win) or -3 (for p2 to win)
        temp = [current_board[0,0], current_board[1,1], current_board[2,2]]
        if sum(temp) == 3:
            return 1
        elif sum(temp) == -3:
            return -1

        temp = [current_board[0,2], current_board[1,1], current_board[2,0]]
        if sum(temp) == 3:
            return 1
        elif sum(temp) == -3:
            return -1

        #Checking for tie if no possible positions left and since no prior win conditions were met
        openSpots = []
        for x in range(3):
            for y in range(3):
                if current_board[x, y] == 0:
                    openSpots.append((x, y))
        if len(openSpots) == 0:
            return 0
        
        #if none of the prior conditions were met, then the game isn't over
        return None


    # only when game ends
    def countEndGameStatus(self):
        result = self.winner()
        # assigning win points
        global WIN_COUNT, LOSS_COUNT, TIE_COUNT, TOTAL_GAMES
        if result == 1:     #p1 wins
            WIN_COUNT += 1
        elif result == -1:  #p2 wins
            LOSS_COUNT += 1
        else:   #tie
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
            # if i % 1000 == 0:
            print(f"Rounds simulated: {i}")
            while not self.gameEnd:
                print(self.board)
                print(self.p1.weights)
                positions = self.availableSpots()
                if self.whoseTurn == 1:
                    action = self.p1.chooseAction(positions, self.board, self.whoseTurn)
                else:
                    action = self.p2.chooseAction(positions, self.board, self.whoseTurn)
                # take action and upate board state
                self.updateState(action)

                win = self.winner()
                if win is not None:
                    self.countEndGameStatus()
                    # ended with p1 either win or draw
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break


class AI:
    def __init__(self, name, learning_rate=0.1):
        self.name = name
        self.learning_rate = learning_rate  # determine rate at which the AI learns
        self.weights = [0.5, 0.5, 0.5] # initialize to a list of 3 zeroes


    # reset board
    def reset(self):
        self.states = []


    # append a hash state
    def addState(self, state, current_board, current_turn):
        self.states.append(state)
        current_board[position] = current_turn
        if current_turn == 1:
            current_turn = -1 
        else:
            current_turn = 1
        self.calculateFeaturesValues()
        self.calculateV_train(self.board.copy(), current_turn)
        self.p1.updateWeights(self.features, self.V_train)


    # determines the value of the board features given a board
    def calculateFeaturesValues(self, current_board):
        features = [0 for i in range(3)]
        features[2] = current_board[1,1] # features[2] tracks who, if anyone, has taken the center space
        for i in range(3):
            # features[0] is the total number of rows, columns, and diagonals that have 2 of Player 1's spaces and 1 open space
            # features[1] is the total number of rows, columns, and diagonals that have 2 of Player 2's spaces and 1 open space
            temp = [current_board[i,0], current_board[i,1], current_board[i,2]] # checks each column
            if sum(temp) == 2: # if the sum is 2, then there has to be two 1s and a 0. No other combination could make this
                features[0] += 1
            if sum(temp) == -2: # if the sum is -2, then there has to be two -1s and a 0. No other combination could make this
                features[1] += 1
            temp = [current_board[0,i], current_board[1,i], current_board[2,i]] # checks each row
            if sum(temp) == 2:
                features[0] += 1
            if sum(temp) == -2:
                features[1] += 1
        temp = [current_board[0,0], current_board[1,1], current_board[2,2]] # checks upper left diagonal
        if sum(temp) == 2:
            features[0] += 1
        if sum(temp) == -2:
            features[1] += 1
        temp = [current_board[2,0], current_board[1,1], current_board[0,2]] # checks upper right diagonal
        if sum(temp) == 2:
            features[0] += 1
        if sum(temp) == -2:
            features[1] += 1
        return features


    # Calculate V^
    def calculateV_hat(self, features):
        V_hat = 0
        for i in range(len(self.weights)):
            V_hat += self.weights[i] * features[i] # Calculate V_hat = w1x1 + w2x2 + w3x3
        return V_hat


    # Calculate a new value for each of the weights
    def updateWeights(self, features, V_train):
        V_hat = self.calculateV_hat(features)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learning_rate * (V_train - V_hat) * features[i]


    # determine AI action
    def chooseAction(self, positions, current_board, current_turn):
        possible_moves = []
        for i in range(3):
            for j in range(3):
                if current_board[i,j] == 0:
                    possible_moves.append((i,j))
        if len(possible_moves) == 1:
            return possible_moves[0]
        else:
            best_move = None
            best_V_hat = 0

            for move in possible_moves:
                if best_move is None:
                    next_board = current_board.copy()
                    next_board[move] = current_turn
                    best_V_hat = self.calculateV_hat(self.calculateFeaturesValues(next_board))
                    best_move = move
                else:
                    next_board = current_board.copy()
                    next_board[move] = current_turn
                    if self.calculateV_hat(self.calculateFeaturesValues(next_board)) > best_V_hat:
                        best_move = move
                        best_V_hat = self.calculateV_hat(self.calculateFeaturesValues(next_board))
            
            return best_move


if __name__ == "__main__":
    # training
    p1 = AI("p1")
    p2 = AI("p2")

    st = GameState(p1, p2)
    training = input("How many matches do you want the AI to self-train: ")
    while not training.isnumeric():
        print("That is not a number. Try again.\n")
        training = input("How many matches do you want the AI to self-train: ")


    print("training...")
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
