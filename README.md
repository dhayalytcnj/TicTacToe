# TicTacToe
TicTacToe learning program for MachineLearning

CONTENTS:
* README.md          - The readme for the project as a markdown file
* main.py            - The source code of the project as a python file
* project1_D1.pdf    - Deliverable 1 as a PDF
* project1_D4.pdf    - Deliverable 4 as a PDF


How to Run:
- Open the terminal
- Navigate to the location where the CJ_YD_Prj1.tar.gz file was downloaded
    Ex. if downloaded to Downloads, enter "cd Downloads"
- To extract the CJ_YD_Prj1.tar.gz file, type "tar -zxvf CJ_YD_Prj1.tar.gz". 
- Type "cd Project1" to get into the extracted folder named Project1 containing the python code
    NOTE: To run the following scripts, make sure you have python3 installed. Enter "sudo apt-get install python3.6" to install it.
- Type "python3 D?.py", in which you replace '?' with the number corresponding to the deliverable you wish to run, and hit enter
- You will be then prompted to enter a tape of your choosing 
- Type in an input and then press enter. Example inputs for each deliverable are given below
- The program will print either "accept" or "reject" based on the machine and input.

In our board features file you can see our initial conditions set for the tic tac toe program and how we went about in creating it. The main program being built around these conditions and having it run by these rules.

In our Graph pdf you can see all of our graph outcomes for the different games played by our program and see how the outcome was overall. Seeing a line graph of seeing the win vs lose ratio plays out over time.

In our Main python file "main.py" this is our main code for the tic tac toe game. 

To run this file you need to do "cd.main.py" 

When looking at our main code you can see that we start off by setting up our global variables and initialize the lists to store our wins, loses, and tie rate. Then we started creating our features for V train and determineing the value of the board features for program to run efficiently. Being able to determine the taken and empty positions on the board and from there letting the program do its thing. Then writing the code for showing how the game will be played out and how the match will occur. We created a class AI for the computer to self learn and being able to create a new value for each weight. With it then playing out and over time training to become more efficient. You can see this after running our program and after playing more tries you can see the win to lose ratio improve. 
