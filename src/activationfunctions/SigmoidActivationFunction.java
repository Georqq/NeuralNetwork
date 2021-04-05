package activationfunctions;

import org.ejml.simple.SimpleMatrix;

public class SigmoidActivationFunction implements ActivationFunction {
    @Override
    public double f(double x) {
        return 1. / (1. + Math.exp(-x)); // 1 / (1 + e^-x)
    }

    @Override
    public double df(double x) {
        double f = 1. / (1. + Math.exp(-x));
        return f * (1. - f);
    }

    @Override
    public double dfOut(double output) {
        return output * (1. - output);
    }

    @Override
    public SimpleMatrix f(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = 1. / (1. + Math.exp(-m.get(i, j))); // 1 / (1 + e^-x)
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
                double f = 1. / (1. + Math.exp(-m.get(i, j)));
                result[i][j] = f * (1. - f);
            }
        }
        return new SimpleMatrix(result);
    }

    @Override
    public SimpleMatrix dfOut(SimpleMatrix output) {
        int rows = output.numRows();
        int cols = output.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = output.get(i, j);
                result[i][j] = val * (1. - val);
            }
        }
        return new SimpleMatrix(result);
    }
}
