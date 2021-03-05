public class Tutorial {

    public static void main(String [] args) {

        
        char [][] Board = {{' ', '|', ' ', '|', ' '},
                {'-', '+', '-', '+', '-'},
                {' ', '|', ' ', '|', ' '},
                {'-', '+', '-', '+', '-'},
                {' ', '|', ' ', '|', ' '}};

        printBoard( Board);

    }

    public static void printBoard( char[][] Board) {
        for(char[] row : Board) {
            for( char c : row) {
                System.out.print(c);
            }
                System.out.println();
    }
}

}
