public class Main {

    public static void main(String[] args) {
        double[][] d1 = { { 1, 2 }, { 3, 4 } };

        double[][] d2 = { { 5, 6 }, { 7, 8 } };

        Matrix matrix1 = new Matrix(d1);
        Matrix matrix2 = new Matrix(d2);

        System.out.println("Matrix 1:");
        System.out.println(matrix1);
        System.out.println("Matrix 2:");
        System.out.println(matrix2);

        System.out.println("Matrix 1 + Matrix 2:");
        Matrix matrix3 = Matrix.add(matrix1, matrix2);
        System.out.println(matrix3);

        System.out.println("Matrix 1 * Matrix 2:");
        Matrix matrix4 = Matrix.multiply(matrix1, matrix2);
        System.out.println(matrix4);

        System.out.println("Setting matrix2 = matrix2 - matrix1");
        matrix2.subtract(matrix1);
        System.out.println(matrix2);

        System.out.println("The determinant of matrix2:");
        System.out.println(matrix2.det());

        double[][] d5 = new double[][] { { 3, -1, 1 }, { 2, 1, 4 } };

        Matrix matrix5 = new Matrix(d5);

        System.out.println(
            "The ref of the matrix:\n" + matrix5 + "is:\n" + Matrix.ref(matrix5)
        );

        System.out.println("The solution to the system of equation is:");
        System.out.println(Matrix.solveEqu(matrix5)[0]);
        System.out.println(Matrix.solveEqu(matrix5)[1]);
    }
}
