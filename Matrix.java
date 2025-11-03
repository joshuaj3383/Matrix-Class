public class Matrix {

    private final int precision = 3;
    private int rows, columns;
    private double[][] matrix;

    // Constructs an identity matrix given rows and columns
    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;

        matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = (i == j ? 1 : 0);
            }
        }
    }

    // Constructor given an array
    public Matrix(double[][] matrix) {
        rows = matrix.length;
        columns = matrix[0].length;

        this.matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.matrix[i][j] = matrix[i][j];
            }
        }
    }

    // Coonstructor given a matrix
    public Matrix(Matrix matrix) {
        rows = matrix.getRows();
        columns = matrix.getColumns();

        this.matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.matrix[i][j] = matrix.getElement(i, j);
            }
        }
    }

    // Getters
    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public double getElement(int row, int column) {
        if (
            row >= this.rows || column >= this.columns || row < 0 || column < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix indices.");
        return matrix[row][column];
    }

    // Setters
    public void setElement(int row, int column, double num) {
        if (
            row >= this.rows || column >= this.columns || row < 0 || column < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix indices.");
        matrix[row][column] = num;
    }

    public void setMatrix(double[][] matrix) {
        rows = matrix.length;
        columns = matrix[0].length;

        this.matrix = matrix;
    }

    // Neatens up display for numbers (2 decimal points)
    private static double rnd(double num, int precision) {
        return (
            Math.round(num * Math.pow(10, precision)) / Math.pow(10, precision)
        );
    }

    // Returns the Matrix as a string
    @Override
    public String toString() {
        String m = "";

        for (int i = 0; i < rows; i++) {
            m += "[";

            for (int j = 0; j < columns - 1; j++) {
                m += String.format("%5s", rnd(matrix[i][j], precision)) + "  ";
            }

            m +=
                String.format("%5s", rnd(matrix[i][columns - 1], precision)) +
                "]\n";
        }

        return m;
    }

    // Row Operations
    // Preform elementary row operations on a Matrix
    // Switch two rows, multiply one by a constant, and add a multiple of 1 to a multiple of another and set it to the first row
    // The static functions do not change the Matrix itself but instead return the new Matrix
    // While the non static functions do update the matrix itself

    // These are used later on in the rref function

    public void swap(int row1, int row2) {
        if (
            row1 >= rows || row2 >= rows || row1 < 0 || row2 < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix rows.");

        double hold;

        for (int j = 0; j < columns; j++) {
            hold = matrix[row1][j];
            matrix[row1][j] = matrix[row2][j];
            matrix[row2][j] = hold;
        }
    }

    public static Matrix swap(Matrix matrix, int row1, int row2) {
        if (
            row1 >= matrix.getRows() ||
            row2 >= matrix.getRows() ||
            row1 < 0 ||
            row2 < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix rows.");

        Matrix mtrx = new Matrix(matrix);

        mtrx.swap(row1, row2);

        return mtrx;
    }

    public void multRow(int row, double num) {
        if (row >= rows || row < 0) throw new IndexOutOfBoundsException(
            "Invalid matrix row."
        );

        for (int j = 0; j < columns; j++) {
            matrix[row][j] *= num;
        }
    }

    public static Matrix multRow(Matrix matrix, int row, double num) {
        if (
            row >= matrix.getRows() || row < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix row.");

        Matrix mtrx = new Matrix(matrix);

        mtrx.multRow(row, num);

        return mtrx;
    }

    // Takes the linear combination of two rows by setting the first one equal to the
    // itself multiplied by a constant times the second row multiplied by another constant
    public void linComb(int row1, double num1, int row2, double num2) {
        if (
            row1 >= rows || row2 > rows || row1 < 0 || row2 < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix rows.");

        for (int j = 0; j < columns; j++) {
            matrix[row1][j] = num1 * matrix[row1][j] + num2 * matrix[row2][j];
        }
    }

    public static Matrix linComb(
        Matrix matrix,
        int row1,
        double num1,
        int row2,
        double num2
    ) {
        if (
            row1 >= matrix.getRows() ||
            row2 >= matrix.getRows() ||
            row1 < 0 ||
            row2 < 0
        ) throw new IndexOutOfBoundsException("Invalid matrix rows.");

        Matrix mtrx = new Matrix(matrix);

        mtrx.linComb(row1, num1, row2, num2);

        return mtrx;
    }

    // Matrix Operations
    // Scale by a constant, add, subtract, multiply, rref, and calculate the determinant
    // The static functions do not change the Matrix itself but instead return the new Matrix
    // While the non static functions do update the matrix itself

    // Multiplies the matrix by a constant
    public void scale(double num) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] *= num;
            }
        }
    }

    // Return the matrix multiplied by a constant
    public static Matrix scale(Matrix matrix, double num) {
        Matrix mtx = new Matrix(matrix);

        mtx.scale(num);

        return mtx;
    }

    // Adds the second matrix to the first
    public void add(Matrix matrix2) {
        if (
            rows != matrix2.getRows() || columns != matrix2.getColumns()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] += matrix2.getElement(i, j);
            }
        }
    }

    // Returns the sum of the two matrixes
    public static Matrix add(Matrix matrix1, Matrix matrix2) {
        if (
            matrix1.getRows() != matrix2.getRows() ||
            matrix1.getColumns() != matrix2.getColumns()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        Matrix mtx = new Matrix(matrix1);

        mtx.add(matrix2);

        return mtx;
    }

    // Subtracts the second matrix from the first
    public void subtract(Matrix matrix2) {
        if (
            rows != matrix2.getRows() || columns != matrix2.getColumns()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] -= matrix2.getElement(i, j);
            }
        }
    }

    // Returns the second matrix subtracted from the first
    public static Matrix subtract(Matrix matrix1, Matrix matrix2) {
        if (
            matrix1.getRows() != matrix2.getRows() ||
            matrix1.getColumns() != matrix2.getColumns()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        Matrix mtx = new Matrix(matrix1);

        mtx.subtract(matrix2);

        return mtx;
    }

    // Sets the first matrix equal to its product with the second matrix
    // [first][second]
    public void multiply(Matrix matrix2) {
        if (columns != matrix2.getRows()) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        // Result Matrix
        double[][] mtx = new double[rows][matrix2.getColumns()];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < matrix2.getColumns(); j++) {
                // Calculate each element
                for (int x = 0; x < columns; x++) {
                    mtx[i][j] += matrix[i][x] * matrix2.getElement(x, j);
                }
            }
        }

        // Update columns
        columns = matrix2.getColumns();

        // Update matrix
        matrix = mtx;
    }

    // Returns the product of two matrixs [first][second]
    public static Matrix multiply(Matrix matrix1, Matrix matrix2) {
        if (
            matrix1.getColumns() != matrix2.getRows()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        Matrix mtx = new Matrix(matrix1);

        mtx.multiply(matrix2);

        return mtx;
    }

    // Takes the dot product of the two matrix's
    // This is intended for vectors, but can also be done on a matrix
    public Matrix dot(Matrix matrix2) {
        if (
            rows != matrix2.getRows() || columns != matrix2.getColumns()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] *= matrix2.getElement(i, j);
            }
        }

        Matrix mtrx = new Matrix(matrix);

        return mtrx;
    }

    public static Matrix dot(Matrix matrix1, Matrix matrix2) {
        if (
            matrix1.getRows() != matrix2.getRows() ||
            matrix1.getColumns() != matrix2.getColumns()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        Matrix mtx = new Matrix(matrix1);

        mtx.dot(matrix2);

        return mtx;
    }

    // Returns true if two matrix's are equivalent in size and all elements, else false
    public boolean equals(Matrix matrix2) {
        if (
            rows != matrix2.getRows() || columns != matrix2.getColumns()
        ) return false;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                // Checks if the two numbers are within 10^-6 of each other
                if (
                    Math.abs(matrix[i][j] - matrix2.getElement(i, j)) > 1e-6
                ) return false;
            }
        }

        return true;
    }

    // Sets one matrix equal to another
    public void setEqual(Matrix mtrx) {
        rows = mtrx.getRows();
        columns = mtrx.getColumns();

        matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = mtrx.getElement(i, j);
            }
        }
    }

    // This funcition appends one matrix to the end of another
    // They must have the same number of rows
    public static Matrix append(Matrix matrix1, Matrix matrix2) {
        if (
            matrix1.getRows() != matrix2.getRows()
        ) throw new IllegalArgumentException(
            "Matrix dimensions do not match for this operation."
        );

        Matrix mtrx = new Matrix(
            matrix1.getRows(),
            matrix1.getColumns() + matrix2.getColumns()
        );

        // Copy over matrix one first
        for (int i = 0; i < mtrx.getRows(); i++) {
            for (int j = 0; j < matrix1.getColumns(); j++) {
                mtrx.setElement(i, j, matrix1.getElement(i, j));
            }
        }

        // Start where matrix one ended and copy over matrix 2
        for (int i = 0; i < mtrx.getRows(); i++) {
            for (int j = matrix1.getColumns(); j < mtrx.getColumns(); j++) {
                mtrx.setElement(
                    i,
                    j,
                    matrix2.getElement(i, j - matrix1.getColumns())
                );
            }
        }

        return mtrx;
    }

    // Matrix transforms and relation matrixes
    // ref, rref, determinant, inverse,

    // This function transposes the matrix by "reflecting" it over the diagonal
    public static Matrix transpose(Matrix matrix) {
        Matrix mtrx = new Matrix(matrix.getColumns(), matrix.getRows());

        for (int i = 0; i < mtrx.getRows(); i++) {
            for (int j = 0; j < mtrx.getColumns(); j++) {
                mtrx.setElement(i, j, matrix.getElement(j, i));
            }
        }

        return mtrx;
    }

    // This returns the transpose of a matrix
    public Matrix transpose() {
        // create a copy of matrix data
        double matrixData[][] = matrix;

        // swap rows and columns
        int holdRow = rows;
        rows = columns;
        columns = holdRow;

        matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = matrixData[j][i];
            }
        }

        Matrix mtrx = new Matrix(matrix);

        return mtrx;
    }

    // ref a matrix while keeping determinant equivalency
    public static Matrix ref(Matrix mtrx) {
        Matrix matrix = new Matrix(mtrx);

        // Last row has no other rows to modify
        for (int row1 = 0; row1 < matrix.getRows() - 1; row1++) {
            // Eleminate all rows below it
            for (int row2 = row1 + 1; row2 < matrix.getRows(); row2++) {
                // Preserves determinant by not scaling the row being changed
                matrix.linComb(
                    row2,
                    1,
                    row1,
                    (-1 * matrix.getElement(row2, row1)) /
                        matrix.getElement(row1, row1)
                );
            }
        }

        return matrix;
    }

    public Matrix ref() {
        Matrix mtrx = new Matrix(matrix);
        mtrx.setEqual(Matrix.ref(mtrx));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = mtrx.getElement(i, j);
            }
        }

        return mtrx;
    }

    // This returns the matrix in row-reduced echelon form, meaning that it has leading ones in the diagonal
    // and zeros above and below
    // returns a null matrix if this is not possible (dividing by 0)
    // O(n^2 * m)
    public static Matrix rrefOld(Matrix mtrx) {
        int numSwap = 0;
        Matrix matrix = new Matrix(mtrx);

        // Gets 1 along the diagonal and zeros below;
        for (int row1 = 0; row1 < matrix.getRows() - numSwap; row1++) {
            // While the leading element is 0
            // and we are not swapping rows which have been pushed to the end
            while (
                matrix.getElement(row1, row1) == 0 &&
                row1 < matrix.getRows() - numSwap
            ) {
                // Update number of zero rows
                numSwap++;
                // swap the zero-row with the last non-zero row
                matrix.swap(row1, matrix.getRows() - numSwap);
            }

            // As long as the leading element is still zero
            if (matrix.getElement(row1, row1) != 0) {
                // Get the leading element of the row to be 1
                matrix.multRow(row1, 1.0 / matrix.getElement(row1, row1));

                // Eleminate all rows below it
                for (int row2 = row1 + 1; row2 < matrix.getRows(); row2++) {
                    matrix.linComb(
                        row2,
                        1,
                        row1,
                        -1 * matrix.getElement(row2, row1)
                    );
                }
            }
        }

        // Gets 1 along the diagonal and zeros below;
        // Starts at the last row which was sucsessfuly rrefed
        for (int row1 = matrix.getRows() - 1 - numSwap; row1 >= 0; row1--) {
            // Move up the rows from the row above row 1 to the top row and
            // eleminate the elements
            for (int row2 = row1 - 1; row2 >= 0; row2--) {
                matrix.linComb(
                    row2,
                    1,
                    row1,
                    -1 * matrix.getElement(row2, row1)
                );
            }
        }

        return matrix;
    }

    /*
    // Credit to ChatGPT for the idea of a row variable to fix the the old rref
    public static Matrix rrefNew(Matrix mtrx){
        Matrix matrix = new Matrix(mtrx);
        int lead = 0;
        int i;

        for (int row = 0; row < matrix.getRows(); row++){
            // Once the leading column gets past the number of columns, the matrix fully rref
            if (lead >= matrix.getColumns()) return matrix;

            // Initial the starting column to the current row
            i = row;
            while (matrix.getElement(i, lead))



        }
        return matrix;
    }
    */

    public static Matrix rref(Matrix mtrx) {
        Matrix matrix = new Matrix(mtrx);
        int lead = 0;
        int rowCount = matrix.getRows();
        int columnCount = matrix.getColumns();

        for (int r = 0; r < rowCount; r++) {
            if (lead >= columnCount) {
                return matrix;
            }
            int i = r;
            while (matrix.getElement(i, lead) == 0) {
                i++;
                if (i == rowCount) {
                    i = r;
                    lead++;
                    if (lead == columnCount) {
                        return matrix;
                    }
                }
            }
            matrix.swap(i, r);

            double leadValue = matrix.getElement(r, lead);
            if (leadValue != 0) {
                matrix.multRow(r, 1.0 / leadValue);
            }

            for (int j = 0; j < rowCount; j++) {
                if (j != r) {
                    double factor = matrix.getElement(j, lead);
                    matrix.linComb(j, 1, r, -factor);
                }
            }
            lead++;
        }
        return matrix;
    }

    // Sets the matrix to its rref form
    public Matrix rref() {
        Matrix mtrx = new Matrix(matrix);
        mtrx.setEqual(Matrix.rref(mtrx));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = mtrx.getElement(i, j);
            }
        }

        return mtrx;
    }

    // Returns the subMatrix of a given matrix
    // This matrix is 1 row and column small because
    // The corresponding row and column where removed
    public Matrix subMatrix(int row, int column) {
        Matrix mtrx = new Matrix(rows - 1, columns - 1);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                // Copies over all numbers except the ones the specified row or column
                // If the current row is greater than the removed row, subtract one
                if (i != row && j != column) mtrx.setElement(
                    (i > row ? i - 1 : i),
                    (j > column ? j - 1 : j),
                    matrix[i][j]
                );
            }
        }

        return mtrx;
    }

    // Calculates the determinant through recursion
    // Very inefficent as it has O(n!)
    public double detClassic() {
        // You can only take the determinant of a square matrix
        if (rows != columns) throw new IllegalArgumentException(
            "Can not take the determinant of a non-square matrix."
        );

        // Formula for a 2x2 matrix
        if (rows == 2) return (
            matrix[1][1] * matrix[0][0] - matrix[1][0] * matrix[0][1]
        );

        double det = 0;

        // Copy matrix to mtrx
        Matrix mtrx = new Matrix(matrix);

        for (int i = 0; i < rows; i++) {
            det +=
                matrix[i][0] *
                mtrx.subMatrix(i, 0).det() *
                (i % 2 == 0 ? 1 : -1);
        }

        return rnd(det, 10);
    }

    // Applies ref to LU factorize the matrix and takes the determinant by multiplying the diagonal row
    // O(n^2) time complexity
    public double det() {
        if (rows != columns) throw new IllegalArgumentException(
            "Can not take the determinant of a non-square matrix."
        );

        Matrix mtrx = new Matrix(matrix);

        mtrx.ref();

        // The determinant is equal to the product of the columns
        double det = 1;
        for (int i = 0; i < rows; i++) det *= mtrx.getElement(i, i);

        return rnd(det, 10);
    }

    // Calculates and returns the inverse of the matrix through rref
    // O(n^2)
    public Matrix inverse() {
        if (rows != columns) throw new IllegalArgumentException(
            "Can not take the inverse of a non-square matrix."
        );

        // Creates a copy of current matrix
        Matrix matrix1 = new Matrix(matrix);
        // Identity matrix
        Matrix matrix2 = new Matrix(rows, columns);

        if (matrix1.det() == 0) return null; // Determinant can not be 0

        // Append the identity matrix to the current matrix
        Matrix mtrx = new Matrix(Matrix.append(matrix1, matrix2));

        mtrx.rref();

        // Declare a new matrix and set it equal to the second half of the rref matrix
        Matrix inv = new Matrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                inv.setElement(i, j, mtrx.getElement(i, j + columns));
            }
        }

        return inv;
    }

    // Calulates the adjoint of a square Matrix
    // [inverse] = 1/det * [adjoint]
    public static Matrix adjoint(Matrix matrix) {
        if (
            matrix.getRows() != matrix.getColumns()
        ) throw new IllegalArgumentException(
            "Can not take the adjoint of a non-square matrix."
        );

        Matrix mtrx = new Matrix(matrix);

        for (int i = 0; i < mtrx.getRows(); i++) {
            for (int j = 0; j < mtrx.getColumns(); j++) {
                mtrx.setElement(
                    i,
                    j,
                    mtrx.subMatrix(i, j).det() * (i + (j % 2) == 0 ? 1 : -1)
                );
            }
        }

        return mtrx.transpose();
    }

    public static Matrix solveNum(Matrix A, Matrix B) {
        if (A == null || B == null) throw new IllegalArgumentException(
            "Matrix can not be null."
        );
        if (A.getRows() != A.getColumns()) throw new IllegalArgumentException(
            "Matrix A must be a square matrix."
        );
        if (B.getColumns() != 1) throw new IllegalArgumentException(
            "Matrix B must be a vector (have 1 column)."
        );

        Matrix sol = new Matrix(1, A.getColumns());

        Matrix rref = new Matrix(append(A, B).rref());

        // Check for infinite / no solutions
        for (int i = 0; i < A.getRows(); i++) {
            for (int j = 0; j < A.getColumns(); j++) {
                // If the a non-diagon and non solution element is 0, there are infinite solutions
                if (
                    i != j && rref.getElement(i, j) != 0
                ) throw new IllegalArgumentException(
                    "There are infinite solutions."
                );
                // If the row is all zero and but the last column, then there is a contradiction and no solution
                if (
                    i == j &&
                    rref.getElement(i, j) == 0 &&
                    rref.getElement(i, rref.getColumns() - 1) != 0
                ) throw new IllegalArgumentException("There are no solutions.");
            }
        }

        for (int row = A.getRows() - 1; row >= 0; row--) {
            sol.setElement(row, 1, rref.getElement(row, rref.getColumns() - 1));
        }

        return sol;
    }

    // Returns the name of the variable given its position in the matrix
    // and total number of variabless
    private static String getVar(int varNum, int totalVar) {
        if (totalVar > 3) return "x_" + varNum;
        if (varNum == 3) return "z";
        if (varNum == 2) return "y";
        if (varNum == 1) return "x";
        return "Error";
    }

    private static int getLead(Matrix matrix, int row) {
        if (row >= matrix.getRows()) throw new IllegalArgumentException(
            "Rows out of bound."
        );

        for (int i = 0; i < matrix.getColumns(); i++) if (
            matrix.getElement(row, i) != 0
        ) return i;

        return -1;
    }

    public static String[] solveEqu(Matrix A, Matrix B) {
        if (A == null || B == null) throw new IllegalArgumentException(
            "Matrix can not be null"
        );

        if (B.getColumns() != 1) throw new IllegalArgumentException(
            "Matrix B must be a vector (have 1 column)"
        );

        String[] sol = new String[A.getColumns()];

        Matrix rref = new Matrix(append(A, B).rref());

        int count, lead;
        double num;

        // Check if there is a solution to system
        for (int i = 0; i < rref.getRows(); i++) {
            count = 0;

            // For every elemen that is zero, count++
            for (int j = 0; j < rref.getColumns() - 1; j++) {
                if (rref.getElement(i, j) == 0) count++;
            }

            // if the last elemement is not zero, count++
            if (rref.getElement(i, rref.getColumns() - 1) != 0) count++;

            // If all of these conditions are met, then there is a contradiction and no solutiton
            if (count == rref.getColumns()) throw new IllegalArgumentException(
                "There is no solution to this system"
            );
        }

        //for (int i = 0; i < A.getColumns(); i++) sol[i] = getVar(i + 1, A.getColumns()) + " = ";

        // Loop through the rows, find the leading element and update the value for the corresponding variable
        for (int i = rref.getRows() - 1; i >= 0; i--) {
            // Gets the leading number
            lead = getLead(rref, i);

            if (0 <= lead && lead < (rref.getColumns() - 1)) {
                // If the leading number is in range and the last element is not 0
                // Add the last element to the correspoding solution
                if (rref.getElement(i, rref.getColumns() - 1) != 0) sol[lead] =
                    Double.toString(rref.getElement(i, rref.getColumns() - 1));

                // For the numbers between the lead and the last element, add -1 * their corresponding var to the solution
                for (int j = rref.getColumns() - 2; j > lead; j--) {
                    num = rnd(rref.getElement(i, j), 2);

                    sol[lead] += (num > 0 ? " - " : (num < 0 ? " + " : ""));
                    sol[lead] += (num == 0
                        ? ""
                        : Math.abs(num) + getVar(j + 1, rref.getColumns() - 1));
                }
            }
        }

        for (int i = 0; i < rref.getColumns() - 1; i++) {
            if (sol[i] == null) sol[i] =
                getVar(i + 1, rref.getColumns() - 1) +
                " = " +
                getVar(i + 1, rref.getColumns() - 1);
            else sol[i] = getVar(i + 1, rref.getColumns() - 1) + " = " + sol[i];
        }

        return sol;
    }

    public static String[] solveEqu(Matrix mtrx) {
        if (mtrx.getColumns() < 2) throw new IllegalArgumentException(
            "System must have at least 2 columns."
        );

        String[] sol = new String[mtrx.getColumns() - 1];

        Matrix rref = new Matrix(mtrx.rref());

        int count, lead;
        double num;

        // Check if there is a solution to system
        for (int i = 0; i < rref.getRows(); i++) {
            count = 0;

            // For every elemen that is zero, count++
            for (int j = 0; j < rref.getColumns() - 1; j++) {
                if (rref.getElement(i, j) == 0) count++;
            }

            // if the last elemement is not zero, count++
            if (rref.getElement(i, rref.getColumns() - 1) != 0) count++;

            // If all of these conditions are met, then there is a contradiction and no solutiton
            if (count == rref.getColumns()) {
                sol[i] = "The are no solutions";
                return sol;
            }
        }

        //for (int i = 0; i < A.getColumns(); i++) sol[i] = getVar(i + 1, A.getColumns()) + " = ";

        // Loop through the rows, find the leading element and update the value for the corresponding variable
        for (int i = rref.getRows() - 1; i >= 0; i--) {
            // Gets the leading number
            lead = getLead(rref, i);

            if (0 <= lead && lead < (rref.getColumns() - 1)) {
                // If the leading number is in range and the last element is not 0
                // Add the last element to the correspoding solution
                if (rref.getElement(i, rref.getColumns() - 1) != 0) sol[lead] =
                    Double.toString(
                        rnd(rref.getElement(i, rref.getColumns() - 1), 3)
                    );

                // For the numbers between the lead and the last element, add -1 * their corresponding var to the solution
                for (int j = rref.getColumns() - 2; j > lead; j--) {
                    num = rnd(rref.getElement(i, j), 3);

                    sol[lead] += (num > 0 ? " - " : (num < 0 ? " + " : ""));
                    sol[lead] += (num == 0
                        ? ""
                        : rnd(Math.abs(num), 3) +
                          getVar(j + 1, rref.getColumns() - 1));
                }
            }
        }

        for (int i = 0; i < rref.getColumns() - 1; i++) {
            if (sol[i] == null) sol[i] =
                getVar(i + 1, rref.getColumns() - 1) +
                " = " +
                getVar(i + 1, rref.getColumns() - 1);
            else sol[i] = getVar(i + 1, rref.getColumns() - 1) + " = " + sol[i];
        }

        return sol;
    }

    public static String[] systemToString(Matrix matrix) {
        double num;
        String[] sol = new String[matrix.getRows()];

        if (matrix.getColumns() < 2) throw new IllegalArgumentException(
            "System must have at least 2 columns."
        );

        for (int i = 0; i < matrix.getRows(); i++) {
            int k = 0;
            while (k < matrix.getColumns() && matrix.getElement(i, k) == 0) k++;

            if (k < matrix.getColumns()) {
                num = matrix.getElement(i, k);
                sol[i] = rnd(num, 3) + getVar(k + 1, matrix.getColumns() - 1);

                for (int j = k + 1; j < matrix.getColumns() - 1; j++) {
                    num = matrix.getElement(i, j);

                    sol[i] += (num > 0 ? " + " : (num < 0 ? " - " : ""));
                    sol[i] += (num == 0
                        ? ""
                        : rnd(Math.abs(num), 3) +
                          getVar(j + 1, matrix.getColumns() - 1));
                }

                sol[i] +=
                    " = " +
                    rnd(matrix.getElement(i, matrix.getColumns() - 1), 3);
            } else sol[i] = "0 = 0";
        }

        return sol;
    }
}
