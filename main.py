
'''
Yash Dhayal, Michael Giordano, Corbin Grosso, & Yuriy
CSC 426-01

Project 1: Self-Learning Tic Tac Toe
'''

import numpy

#global variables for win, loss, and tie counts
WIN_COUNT = 0
LOSS_COUNT = 0
TIE_COUNT = 0

class State:
    #initializing some variables
    def __init__(self, p1, p2):
        self.board = numpy.zeros((3, 3))  #3x3 board with just zeroes
        self.p1 = p1
        self.p2 = p2
        self.gameEnd = False    # check if game is over
        self.boardHash = None
        self.whoseTurn = 1      # init p1 plays first


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


    # only when game ends
    def winPoints(self):
        result = self.winner()
        # assigning win points
        global WIN_COUNT, LOSS_COUNT, TIE_COUNT
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


    # How a match will occur
    def play(self, rounds):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds simulated: {}".format(i))
            while not self.gameEnd:
                # AI 1
                positions = self.availableSpots()
                p1_action = self.p1.chooseAction(positions, self.board, self.whoseTurn)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.winPoints()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # AI 2
                    positions = self.availableSpots()
                    p2_action = self.p2.chooseAction(positions, self.board, self.whoseTurn)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.winPoints()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break


class AI:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value


    # reset board
    def reset(self):
        self.states = []


    def getHash(self, board):
        boardHash = str(board.reshape(3 * 3))
        return boardHash


    # append a hash state
    def addState(self, state):
        self.states.append(state)


    # determine AI action
    def chooseAction(self, positions, current_board, symbol):
        if numpy.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            action = positions[numpy.random.choice(len(positions))]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
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

    st = State(p1, p2)
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


