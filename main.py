
'''
Yash Dhayal, Michael Giordano, Corbin Grosso, & Yuriy
CSC 426-01
3/3/2021

Project 1: Self-Learning Tic Tac Toe
'''

import numpy

class State:
    def __init__(self, p1, p2):
        self.board = numpy.zeros(3, 3)  #3x3 board with just zeroes
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.whoseTurn = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(3 * 3))
        return self.boardHash

    def winner(self):
        # Checking if sum of rows = 3 (for p1 to win) or -3 (for p2 to win)
        for x in range(3):
            if sum(self.board[x, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[x, :]) == -3:
                self.isEnd = True
                return -1
        
        # Checking if sum of columns = 3 (for p1 to win) or -3 (for p2 to win)
        for x in range(3):
            if sum(self.board[:, x]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, x]) == -3:
                self.isEnd = True
                return -1

        # Checking if sum of diagonals = 3 (for p1 to win) or -3 (for p2 to win)
        diag_sum1 = sum([self.board[i, i] for i in range(3)])
        diag_sum2 = sum([self.board[i, 3 - i - 1] for i in range(3)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        #Checking for tie if no possible positions left and since no prior win conditions were met
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        
        #if none of the prior conditions were met, then the game isn't over
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for x in range(3):
            for y in range(3):
                if self.board[x, y] == 0:
                    positions.append((x, y))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.whoseTurn
        # switch to another player
        self.whoseTurn = -1 if self.whoseTurn == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = numpy.zeros((3, 3))
        self.boardHash = None
        self.isEnd = False
        self.whoseTurn = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # AI 1
                positions = self.availablePositions()
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
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # AI 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.whoseTurn)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break


    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class AI:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(3 * 3))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if numpy.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = numpy.random.choice(len(positions))
            action = positions[idx]
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

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
            fw = open('save.txt', 'w+')
            .dump(self.states_value, fw)
            fw.close()


if __name__ == "__main__":
    # training
    p1 = AI("p1")
    p2 = AI("p2")

    st = State(p1, p2)
    print("training...")
    st.play(3000)


