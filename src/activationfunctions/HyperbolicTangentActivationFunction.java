package activationfunctions;

import org.ejml.simple.SimpleMatrix;

public class HyperbolicTangentActivationFunction implements ActivationFunction {
    @Override
    public double f(double x) {
        double exp = Math.exp(x);
        double negExp = Math.exp(-x);
        return (exp - negExp) / (exp + negExp);
    }

    @Override
    public double df(double x) {
        double exp = Math.exp(x);
        double negExp = Math.exp(-x);
        double tanh = (exp - negExp) / (exp + negExp);
        return 1. - tanh * tanh;
    }

    @Override
    public double dfOut(double x) {
        return 1. - x * x;
    }

    @Override
    public SimpleMatrix f(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = m.get(i, j);
                double exp = Math.exp(x);
                double negExp = Math.exp(-x);
                result[i][j] = (exp - negExp) / (exp + negExp);
            }
        }
        return new SimpleMatrix(result);
    }

    @Override
    public SimpleMatrix df(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = m.get(i, j);
                double exp = Math.exp(x);
                double negExp = Math.exp(-x);
                double tanh = (exp - negExp) / (exp + negExp);
                result[i][j] = 1. - tanh * tanh;
            }
        }
        return new SimpleMatrix(result);
    }

    @Override
    public SimpleMatrix dfOut(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = m.get(i, j);
                result[i][j] = 1. - (x * x);
            }
        }
        return new SimpleMatrix(result);
    }
}
