package activationfunctions;

import org.ejml.simple.SimpleMatrix;

public class ReLUActivationFunction implements ActivationFunction{
    @Override
    public double f(double x) {
        return x >= 0 ? x : 0.;
    }

    @Override
    public double df(double x) {
        return x >= 0 ? 1. : 0.;
    }

    @Override
    public SimpleMatrix f(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = m.get(i, j);
                result[i][j] = val >= 0 ? val : 0.;
            }
        }
        return new SimpleMatrix(result);
    }

    @Override
    public double dfOut(double output) {
        return output >= 0 ? 1. : 0.;
    }

    @Override
    public SimpleMatrix df(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = m.get(i, j);
                result[i][j] = val >= 0 ? 1. : 0.;
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
                result[i][j] = val >= 0 ? 1. : 0.;
            }
        }
        return new SimpleMatrix(result);
    }
}
