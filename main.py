'''
Yash Dhayal, Michael Giordano, Corbin Grosso, & Yuriy Deyneka
CSC 426-01
Project 1: Self-Learning Tic Tac Toe
'''

import numpy
import matplotlib.pyplot as plt # used to get graphs
import random

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

TURN = 1 # tracks which AI is currently making a move
class GameState:
    """
    Class containing data about the game board and the two AIs that are playing playing
    """
    
    def __init__(self, p1, p2):
        """
        Initializes variables for GameState class objects

        :param p1: AI player 1, the AI that is being actively trained
        :param p2: AI player 2, the opponent of the AI that is being actively trained
        :type p1: AI class object
        :type p2: AI class object
        """

        self.board = numpy.zeros((3, 3))  #3x3 board with just zeroes
        self.p1 = p1
        self.p2 = p2
        self.gameEnd = False    # check if game is over
        self.features = [0 for i in range(3)]      # init to a list of three zeroes; tracks the boards current features
        self.V_train = 0    # init to 0; will be calculated before use


    def updateState(self, position):
        """
        Modifies the board to claim a space for a player and switches which AI's turn it is

        :param position: Position that a player has just taken on the board
        :type position: numpy array
        """

        global TURN
        self.board[position] = TURN
        if TURN == 1:
            TURN = -1 
        else:
            TURN = 1
        self.p1.calculateFeaturesValues(self.board)
        self.calculateV_train(self.board.copy(), TURN)
        self.p1.updateWeights(self.features, self.V_train)


    def calculateV_train(self, current_board, turn): 
        """
        Calculates the value of the V_train variable by recursively checking every possible way the game could play out
        from the board's current state

        :param current_board: the current state of the board
        :param turn: which AI is currently making a move in the game. Global variable TURN can not be used due to recursion
        :type current_board: numpy array
        :type turn: int
        """

        current_turn = turn
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
            self.calculateV_train(current_board.copy(), -current_turn)
        else:
            return win


    def reset(self):
        """
        Prepares the state of the board for a new game
        """

        self.board = numpy.zeros((3, 3))
        self.gameEnd = False
        self.features = [0 for i in range(3)]
        TURN = 1


    def availableSpots(self):
        """
        Creates and returns a list of all empty spaces currently on the board
        """

        openSpots = []
        for x in range(3):
            for y in range(3):
                if self.board[x, y] == 0:
                    openSpots.append((x, y))  # need to be tuple
        return openSpots


    def winner(self):
        """
        Checks if the game has been won by either AI and returns the winner (or 0 for a draw). 
        Returns none if there is not yet a winner.
        Also changes the gameEnd variable as needed
        """

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


    def winnerPredict(self, current_board):
        """
        Functions the same as the winner function, but does not end the game when a winner is found.
        This function is used as a helper to calculate V_train

        :param current_board: board state to be looked at
        :type current_board: numpy array
        """
        
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


    def countEndGameStatus(self):
        """
        Tracks the wins, losses, ties, and total number of games between the two AIs during training.
        This is used at the end to generate the graphs of the AI's performance
        """

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


    def play(self, rounds):
        """
        Simulates games of Tic-Tac-Toe between two opposing AIs

        :param rounds: number of games of Tic-Tac-Toe to simulate
        :type rounds: int
        """

        for i in range(rounds):
            if i % 100 == 0:
                print(f"Rounds simulated: {i}")
            while not self.gameEnd:
                positions = self.availableSpots()
                if TURN == 1:
                    action = self.p1.chooseAction(positions, self.board, TURN)
                else:
                    action = self.p2.chooseAction(positions, self.board, TURN)
                # take action and upate board state
                self.updateState(action)

                win = self.winner()
                if win is not None: # If the game has ended
                    self.countEndGameStatus()
                    self.reset()
                    break


class AI:
    """
    Class containing data about an AI to play Tic-Tac-Toe
    """

    def __init__(self, name, learning_rate=0.1):
        """
        Initializes variables for an AI class object

        :param name: identifier for the AI
        :param learning_rate: rate at which the AI learns
        :type name: str
        :type learning_rate: float
        """

        self.name = name
        self.learning_rate = learning_rate  # determine rate at which the AI learns
        self.weights = [1 for i in range(3)] # initialize to a list of 3 zeroes


    def calculateFeaturesValues(self, current_board):
        """
        Calculates the values of each of the board features being tracked.
        features[0] is the number of rows/columns/diagonals that contain two 1s and a 0
        features[1] is the number of rows/columns/diagonals that contain two -1s and a 0
        features[2] is the player who has claimed it, 1 for the AI being trained, -1 for the opposing AI
        """

        features = [0 for i in range(3)]
        features[2] = current_board[1,1]
        for i in range(3):
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


    def calculateV_hat(self, features):
        """
        Calculates V_hat based on the provided features

        :param features: features of a board state (explained in detail in the calculateFeatureValues() documentation)
        :type features: list(int)
        """

        V_hat = 0
        for i in range(len(self.weights)):
            V_hat += self.weights[i] * features[i] # Calculate V_hat = w1x1 + w2x2 + w3x3
        return V_hat


    def updateWeights(self, features, V_train):
        """
        Recalculate the value of each of the weights using the provided features and V_train value

        :param features: features of a board state (explained in detail in the calculateFeatureValues() documentation)
        :param V_train: value of the board state determined by the training data
        :type features: list(int)
        :type V_train: int
        """

        V_hat = self.calculateV_hat(features)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learning_rate * (V_train - V_hat) * features[i]


    # determine AI action
    def chooseAction(self, positions, current_board, current_turn):
        """
        Decides and returns the best possible move choice available in the form of coordinates to the space to take

        :param positions: list of open spaces on the board
        :param current_board: Current state of the board
        :param current_turn: which AI is currently making a move
        :type positions: list(numpy array)
        :type current_board: numpy array
        :type current_turn: int
        """

        possible_moves = []
        for i in range(3):
            for j in range(3):
                if current_board[i,j] == 0:
                    possible_moves.append((i,j))
        if len(possible_moves) == 1:
            return possible_moves[0]
        elif len(possible_moves) == 9 and current_turn == -1:
            return possible_moves[int(random.randint(0, 8))]
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
    generate_graphs = input("Would you like to generate new graphs (located in the project1_D3 files) " + 
    "based on the newly generaed data?" + 
    "\nWARNING: This will overwrite those files if they already exist in the working directory." +
    "\nEnter 1 for yes or 0 for no.")
    while not generate_graphs.isnumeric() or int(generate_graphs) not in [0, 1]:
        print('That is not a valid response.')
        generate_graphs = input("Would you like to generate new graphs (located in the project1_D3 files)" + 
        "based on the newly generaed data?" + 
        "\nWARNING: This will overwrite those files if they already exist in the working directory." +
        "\nEnter 1 for yes or 0 for no: ")

    if generate_graphs == 1:
        plt.plot(totalTracker, winRate)
        plt.title('Win Tracker')
        plt.xlabel('Number of Games Played')
        plt.ylabel('Win Percentage')
        plt.ylim([-1, 101])
        plt.xlim([1, TOTAL_GAMES])
        plt.savefig('project1_D3_win.pdf')
        plt.clf()

        plt.plot(totalTracker, loseRate)
        plt.title('Lose Tracker')
        plt.xlabel('Number of Games Played')
        plt.ylabel('Lose Percentage')
        plt.ylim([-1, 101])
        plt.xlim([1, TOTAL_GAMES])
        plt.savefig('project1_D3_lose.pdf')
        plt.clf()

        plt.plot(totalTracker, tieRate)
        plt.title('Tie Tracker')
        plt.xlabel('Number of Games Played')
        plt.ylabel('Draw Percentage')
        plt.ylim([-1, 101])
        plt.xlim([1, TOTAL_GAMES])
        plt.savefig('project1_D3_tie.pdf')
        plt.clf()
