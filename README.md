# Matrix-Class
This is a matrix class for java which provides the following capabilities:

1. Elemental Row Operations: Swaping, scaling, and making linear combinations out of rows
2. Basic Arithmetic: Matrix addition, Scaling, and Matrix multiplication
3. Transformations: Ref, RREF, and transpose
4. Determinant: Calculated through REF
5. Inverse: Via augmented matrix and REF
6. Solvers: Can solve systems of equations

I wrote this as I was learning java and reviewing linear algebra as a way to apply these two different subjects and look at them through a new perspective.

I would like to note that there are a few design flaws as well as documentation errors as I did not fully understand design principles at this time.
1. The matrix objects are all mutable which is not standard, but isnt awful. If you want to treat them as immutable, you can access all operations through the Matrix class and they're static variation.
2. The solver returns a string which is nice to read, but really should return the solution matrix
3. Used regular comments instead of javadocs

Run the test Main to see some of the capabilities of the class.
