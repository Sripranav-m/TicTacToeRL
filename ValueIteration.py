#CS18B036 SRIPRANAV MANNEPALLI

#import modules
# from numpy.lib.funciton_base import average
import pygame
from pygame.locals import *
import numpy as np
from numpy import linalg as LA

class TicTacToe:
    def __init__(self,dimension=3):
        pygame.init()
        self.defineBoard(dimension)
        self.defineVariables()
        self.runMainLoop()

    # Define all our Board variables
    def defineBoard(self,dimension):
        self.boardDimension = dimension
        self.screenHeight = self.boardDimension * 100
        self.screenWidth = self.boardDimension * 100
        self.lineWidth = 6
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption('Tic Tac Toe')
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.font = pygame.font.SysFont(None, 40)
        self.againRectangle = Rect(self.screenWidth // 2 - 80, self.screenHeight // 2, 160, 50)
        self.markers = []
        #create empty 3 x 3 list to represent the grid
        for x in range (self.boardDimension):
            row = [0] * self.boardDimension
            self.markers.append(row)
        self.markers = np.array(self.markers)
        self.gameBoard = self.markers[:]

    # Define all our other variables required
    def defineVariables(self):
        self.clicked = False
        self.player = 1
        self.pos = (0,0)
        self.gameOver = False
        self.winner = 0
        self.valueIterationPolicy = []
        self.run = True
        self.statespace = []
        # 1 - win, 2 - loss, 3 - draw : terminal states
        # 0 - non-termina state
        self.stateSpaceIdentification = []
        self.statespace.append(0)
        self.stateSpaceIdentification.append(0)
        self.genAllStates(self.gameBoard)
        self.stateSpaceLength = len(self.statespace)
        self.PTF = np.zeros((self.stateSpaceLength, self.boardDimension**2, self.stateSpaceLength))
        self.rewardFunction = np.zeros((self.stateSpaceLength, self.boardDimension**2, self.stateSpaceLength))
        self.gameOver = False
        self.winner = 0
        self.numberOfGames = 1000
        self.wins = 0
        self.losses = 0
        self.ties = 0

    # Draw TicTacToe Board
    def drawBoard(self):
        bg = (255, 255, 210)
        grid = (50, 50, 50)
        self.screen.fill(bg)
        for x in range(1,self.boardDimension):
            pygame.draw.line(self.screen, grid, (0, 100 * x), (self.screenWidth,100 * x), self.lineWidth)
            pygame.draw.line(self.screen, grid, (100 * x, 0), (100 * x, self.screenHeight), self.lineWidth)

    # Draw Markers of TicTacToe Board
    def drawMarkers(self):
        x_pos = 0
        for x in self.markers:
            y_pos = 0
            for y in x:
                if y == 1:
                    pygame.draw.line(self.screen, self.red, (y_pos * 100 + 15, x_pos * 100 + 15), (y_pos * 100 + 85, x_pos * 100 + 85), self.lineWidth)
                    pygame.draw.line(self.screen, self.red, (y_pos * 100 + 85, x_pos * 100 + 15), (y_pos * 100 + 15, x_pos * 100 + 85), self.lineWidth)
                if y == -1:
                    pygame.draw.circle(self.screen, self.green, (y_pos * 100 + 50, x_pos * 100 + 50), 38, self.lineWidth)
                y_pos += 1
            x_pos += 1	

    # Check if the game is over or not
    def checkGameOver(self):
        x_pos = 0
        for x in self.markers:
            #check columns 
            if sum(x) == self.boardDimension:
                self.winner = 1
                self.gameOver = True
            elif sum(x) == self.boardDimension*(-1):
                self.winner = 2
                self.gameOver = True
            #check rows
            row_sum = 0
            for cnt in range(0,self.boardDimension):
                row_sum+= self.markers[cnt][x_pos]
            if row_sum == self.boardDimension:
                self.winner = 1
                self.gameOver = True
            elif row_sum == self.boardDimension*(-1):
                self.winner = 2
                self.gameOver = True
            x_pos += 1
        #check cross
        left_diag_sum = 0
        right_diag_sum = 0
        for cnt in range(0,self.boardDimension):
            left_diag_sum+= self.markers[cnt][cnt]
        c_cnt = self.boardDimension - 1
        r_cnt = 0
        while(r_cnt < self.boardDimension):
            right_diag_sum+= self.markers[r_cnt][c_cnt]
            r_cnt+= 1
            c_cnt-= 1
        if left_diag_sum == self.boardDimension or right_diag_sum == self.boardDimension:
            self.winner = 1
            self.gameOver = True
        elif left_diag_sum == self.boardDimension*(-1) or right_diag_sum == self.boardDimension*(-1):
            self.winner = 2
            self.gameOver = True

        # Check for TIE now
        if self.gameOver == False:
            tie = True
            for row in self.markers:
                for i in row:
                    if i == 0:
                        tie = False
            #if it is a tie, then call game over and set self.winner to 0 (no one)
            if tie == True:
                self.gameOver = True
                self.winner = 0


    # Display on TicTacToe Board who wins
    def drawGameOver(self,winner):

        if winner != 0:
            end_text = "Player " + str(winner) + " wins!"
        elif winner == 0:
            end_text = "You have tied!"

        end_img = self.font.render(end_text, True, self.blue)
        pygame.draw.rect(self.screen, self.green, (self.screenWidth // 2 - 100, self.screenHeight // 2 - 60, 200, 50))
        self.screen.blit(end_img, (self.screenWidth // 2 - 100, self.screenHeight // 2 - 50))

        again_text = 'Play Again?'
        again_img = self.font.render(again_text, True, self.blue)
        pygame.draw.rect(self.screen, self.green, self.againRectangle)
        self.screen.blit(again_img, (self.screenWidth // 2 - 80, self.screenHeight // 2 + 10))


    # Convert n-ary number of base "base" to decimal
    def naryToDecimal(self,arr, base):
        decimal = 0
        power = 1
        for i in range(np.size(arr)):
            decimal+= arr[i]*power
            power*= base
        return decimal


    # Convert 3-Dimensional Board array to decimal
    def boardToDecimal(self,board):
        ternary = board.flatten()
        decimal = 0
        power = 1
        for i in range(np.size(board)):
            decimal+= ternary[i]*power
            power*= 3
        return decimal


    # Convert decimal to 3-Dimensional Board array
    def decimalToBoard(self,decimal):
        ternaryList = []
        ternaryList[:0] = np.base_repr(decimal, base=3)[::-1]
        ternaryList = list(map(int,ternaryList))
        ternaryArray = np.array(ternaryList)
        ternaryArray = np.pad(ternaryArray, (0,(self.boardDimension**2)-np.size(ternaryArray)), 'constant')
        board = ternaryArray.reshape(self.boardDimension,self.boardDimension)
        return board

    # Get zero (empty) places from the board
    def getZeroesInBoard(self,board):
        zeroesPositions = []
        for i in range(self.boardDimension):
            for j in range(self.boardDimension):
                if board[i][j] == 0:
                    zeroesPositions.append([i,j])
        return zeroesPositions


    # Generate states with n+1 O's and n+1 X's from the given board with n O's and n X's
    def generateInitialStates(self,board):
        zeroesPositions = self.getZeroesInBoard(board)
        for pos_x in zeroesPositions:
            board[pos_x[0]][pos_x[1]] = 1
            for pos_o in zeroesPositions:
                if(pos_o != pos_x):
                    board[pos_o[0]][pos_o[1]] = 2
                    decimal = self.boardToDecimal(board)
                    if decimal not in self.statespace:
                        # Add the state to our stateSpace
                        self.statespace.append(decimal)
                        zeroesPositions2 = self.getZeroesInBoard(board)
                        if len(zeroesPositions2) == 1:
                            board[zeroesPositions2[0][0]][zeroesPositions2[0][1]] = 1
                            # Check what type is our state Space : 1 (1 is winner) , 2 (2 is winner) , 3 (tie occured) and 0: game is not yet over
                            self.stateSpaceIdentification.append(self.checkBoardState(board))
                            board[zeroesPositions2[0][0]][zeroesPositions2[0][1]] = 0
                        else:
                            # Check what type is our state Space : 1 (1 is winner) , 2 (2 is winner) , 3 (tie occured) and 0: game is not yet over
                            self.stateSpaceIdentification.append(self.checkBoardState(board))
                    board[pos_o[0]][pos_o[1]] = 0
            board[pos_x[0]][pos_x[1]] = 0


    # generate the states from prev_start_idx to prev_end_idx number of X's or O's 
    def genOtherStates(self,prev_start_idx, prev_end_idx):
        for idx in range(prev_start_idx, prev_end_idx+1):
            board = self.decimalToBoard(self.statespace[idx])
            self.generateInitialStates(board)


    # generate all states for our board
    def genAllStates(self,board):
        prev_start_idx = len(self.statespace)
        self.generateInitialStates(board)
        prev_end_idx = len(self.statespace)-1
        # for even board dimension
        if self.boardDimension % 2 == 0:
            for i in range((self.boardDimension**2)//2 - 2):
                self.genOtherStates(prev_start_idx, prev_end_idx)
                prev_start_idx = prev_end_idx +1
                prev_end_idx = len(self.statespace) -1
        # for odd board dimension
        else:
            for i in range((self.boardDimension**2)//2 - 1):
                self.genOtherStates(prev_start_idx, prev_end_idx)
                prev_start_idx = prev_end_idx +1
                prev_end_idx = len(self.statespace) -1		


    # Generate Probability Transition Function
    def generatePTF(self):
        # Loop for all states in the state space
        for stateIdx in range(len(self.statespace)):
            # Get the board configuation for a given state
            board = self.decimalToBoard(self.statespace[stateIdx])
            zeroesPositions = self.getZeroesInBoard(board)
            checkTerminalermState = self.stateSpaceIdentification[stateIdx]
            # If given state is terminal, then probability to 0 state is '1' irrespective of action
            if checkTerminalermState == 1 or checkTerminalermState == 2 or checkTerminalermState == 3:
                for posX in zeroesPositions:
                    actionIdx = self.naryToDecimal(np.array(posX), self.boardDimension)
                    self.PTF[stateIdx][actionIdx][0] = 1
            else:
                for posX in zeroesPositions:
                    # Put X in this spot
                    board[posX[0]][posX[1]] = 1
                    actionIdx = self.naryToDecimal(np.array(posX), self.boardDimension)
                    cnt = 0
                    nextStateIdxs = []
                    for posO in zeroesPositions:
                        if(posO != posX):
                            # Put O in this spot
                            board[posO[0]][posO[1]] = 2
                            decimal = self.boardToDecimal(board)
                            # Get statespace index of this obtains next state
                            nextStateIdx = self.statespace.index(decimal)
                            nextStateIdxs.append(nextStateIdx)
                            cnt += 1
                            board[posO[0]][posO[1]] = 0
                    for idx in nextStateIdxs:
                        # Equal Probability for all the next possible states
                        self.PTF[stateIdx][actionIdx][idx] = 1/cnt
                        # Get reward. If idx is win : 100 reward, If idx is loss : -100, If idx is tie : 10 and 0 otherwise
                        # We get this from an already computed array self.stateSpaceIdentification
                        self.rewardFunction[stateIdx][actionIdx][idx] = self.getReward(idx)
                    board[posX[0]][posX[1]] = 0


    def getReward(self,nxt_st):
        # Win reward
        if (self.stateSpaceIdentification[nxt_st] == 1):
            return 100
        # Lose reward
        elif (self.stateSpaceIdentification[nxt_st] == 2):
            return -100
        # Tie reward
        elif (self.stateSpaceIdentification[nxt_st] == 3):
            return 10
        # Non Terminal State Reward
        return 0

    # Check state of board
    def checkBoardState(self,board):
        board[board == 2] = -1
        self.gameOver = False
        self.winner = -1
        tie1 = True
        # global gameOver
        # global self.winner
        x_pos = 0
        for x in board:
            # check columns
            if sum(x) == self.boardDimension:
                self.winner = 1
                self.gameOver = True
            elif sum(x) == self.boardDimension*(-1):
                self.winner = 2
                self.gameOver = True
            # check rows
            row_sum = 0
            for cnt in range(0, self.boardDimension):
                row_sum += board[cnt][x_pos]
            if row_sum == self.boardDimension:
                self.winner = 1
                self.gameOver = True
            elif row_sum == self.boardDimension*(-1):
                if self.winner != 1:
                    self.winner = 2
                    self.gameOver = True
            x_pos += 1

        # check cross
        left_diag_sum = 0
        right_diag_sum = 0
        for cnt in range(0, self.boardDimension):
            left_diag_sum += board[cnt][cnt]
        c_cnt = self.boardDimension - 1
        r_cnt = 0
        while(r_cnt < self.boardDimension):
            right_diag_sum += board[r_cnt][c_cnt]
            r_cnt += 1
            c_cnt -= 1

        if left_diag_sum == self.boardDimension or right_diag_sum == self.boardDimension:
            self.winner = 1
            self.gameOver = True
        elif left_diag_sum == self.boardDimension*(-1) or right_diag_sum == self.boardDimension*(-1):
            if self.winner != 1:
                self.winner = 2
                self.gameOver = True

        # check for tie
        if self.gameOver == False:
            for row in board:
                for i in row:
                    if i == 0:
                        tie1 = False
            if tie1 == True:
                self.gameOver = True
                self.winner = 0

        board[board == -1] = 2

        if self.gameOver == False:
            return 0
        elif self.winner == 1:
            return 1
        elif self.winner == 2:
            return 2
        elif self.winner == 0:
            return 3

    # Value Iteration
    def valueIteration(self):
        # Initially start with 0 value for all states in our value functions
        v = np.zeros(len(self.statespace))
        v_new = np.zeros(len(self.statespace))
        cnt = 1
        # Do iterations till abs(LA.norm(v_new) - LA.norm(v)) is greater than 0.0001 (practical)
        while(True):
            print("Value Iteration step : ",cnt)
            cnt += 1
            self.valueIterationPolicy.clear()
            
            # Loop for all states in the State Space
            for curr_st in range(0, len(self.statespace)):
                board = self.decimalToBoard(self.statespace[curr_st])
                zeroesPositions = self.getZeroesInBoard(board)

                max_value = float("-inf")
                best_action = []

                for action in zeroesPositions:
                    # Put X in this position
                    board[action[0]][action[1]] = 1
                    # Get the action
                    actionIdx = self.naryToDecimal(np.array(action), self.boardDimension)
                    avgReward = 0

                    for nxt_action in zeroesPositions:
                        if(nxt_action != action):
                            # Put O in this position
                            board[nxt_action[0]][nxt_action[1]] = 2
                            decimal = self.boardToDecimal(board)
                            nxtState = self.statespace.index(decimal)
                            avgReward += self.PTF[curr_st][actionIdx][nxtState] * self.rewardFunction[curr_st][actionIdx][nxtState]
                            board[nxt_action[0]][nxt_action[1]] = 0
                    gamma_pv_term = np.dot(self.PTF[curr_st][actionIdx], v)
                    t_pi_v_pi = avgReward + 0.1*gamma_pv_term
                    if max_value < t_pi_v_pi:
                        max_value = t_pi_v_pi
                        best_action = action
                    board[action[0]][action[1]] = 0

                v_new[curr_st] = max_value
                # best_action is best for the state
                self.valueIterationPolicy.append(best_action)

            # Break the loop in this case
            if(abs(LA.norm(v_new) - LA.norm(v)) < 0.00001):
                break
            v = v_new
            v_new = np.zeros(len(self.statespace))

    # Run the main loop
    def runMainLoop(self):
        # Generate Probability Transition Function matrix : State x Action x State
        self.generatePTF()
        # Do Value Iteration
        self.valueIteration()
        # loop for self.numberOfGames number of times
        for _ in range(0,self.numberOfGames,1):
            #draw board and markers first
            self.drawBoard()
            self.drawMarkers()
            #handle events
            for event in pygame.event.get():
                #handle game exit
                if event.type == pygame.QUIT:
                    self.run = False
                #run new game
            while self.gameOver == False:
                self.markers[self.markers == -1] = 2
                prsnt_state = self.statespace.index(self.boardToDecimal(self.markers))
                action = self.valueIterationPolicy[prsnt_state] 
                self.markers[action[0]][action[1]] = 1
                self.markers[self.markers == 2] = -1
                self.checkGameOver()
                if self.gameOver == True:
                    break
                self.markers[action[0]][action[1]] = 0
                self.markers[self.markers == -1] = 2
                prsnt_action = self.naryToDecimal(np.array(action), self.boardDimension)
                # Pick the next state randomly from the possible states and their probabilities
                nxt_state_idx = np.random.choice(range(self.stateSpaceLength), 1, p=self.PTF[prsnt_state][prsnt_action])[0]
                self.markers = self.decimalToBoard(self.statespace[nxt_state_idx])
                self.markers[self.markers == 2] = -1
                self.checkGameOver()
                if self.gameOver == True:
                    break

            #check if game has been won
            if self.gameOver == True:
                self.drawGameOver(self.winner)
                if self.winner == 1:
                    self.wins+=1
                elif self.winner == 2:
                    self.losses+=1
                elif self.winner == 0:
                    self.ties+=1
                #reset variables
                self.gameOver = False
                self.player = 1
                pos = (0,0)
                self.markers = []
                self.winner = 0
                #create empty 3 x 3 list to represent the grid
                for x in range (self.boardDimension):
                    row = [0] * self.boardDimension
                    self.markers.append(row)
                self.markers = np.array(self.markers)

            #update display
            pygame.display.update()

        pygame.quit()
        print("==============================================================")
        print("Total Number of Games Played :", self.numberOfGames)
        print("WIN percentage               :", (self.wins/1000)*100)
        print("TIE percentage               :", (self.ties/1000)*100)
        print("LOSS percentage              :", (self.losses/1000)*100)
        print("==============================================================")


t=TicTacToe()