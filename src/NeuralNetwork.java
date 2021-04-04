import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    SimpleMatrix weightsHidden;
    SimpleMatrix weightsOut;
    SimpleMatrix hiddenOut;
    SimpleMatrix out;
    SimpleMatrix biasHidden;
    SimpleMatrix biasOut;

    double res;
    double meanErr;

    double learningRate = 1.;

    static double [][] X = {
            {0., 0.},
            {1., 0.},
            {0., 1.},
            {1., 1.}
    };
    static double [][] Y = {
            {0.},{1.},{1.},{0.}
    };
    static double[][] testX = {
            {0., 0.},
            {0., 1.},
            {1., 0.},
            {1., 1.}
    };
    static double[][] testY = {
            {0.},{1.},{1.},{0.}
    };

    public static void main(String[] args) {
        long t1 = System.nanoTime();
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 5, 1);
        neuralNetwork.train(X, Y, 1_000_001);
        long t2 = System.nanoTime();
        System.out.println("Time: " + (t2 - t1) / 1_000_000 + " ms");
        neuralNetwork.test(testX, testY, true);
    }

    private void test(double[][] testX, double[][] testY, boolean output) {
        meanErr = 0.;
        for (int i = 0; i < testX.length; i++) {
            check(testX[i], testY[i]);
        }
        if (output) {
            System.out.println("MeanError: " + meanErr);
        }
    }

    public NeuralNetwork(int input, int hidden, int output) {
        weightsHidden = new SimpleMatrix(hidden, input);
        weightsOut = new SimpleMatrix(output, hidden);
        biasHidden = new SimpleMatrix(hidden, 1);
        biasOut = new SimpleMatrix(output, 1);
        hiddenOut = new SimpleMatrix(hidden , 1);
        out = new SimpleMatrix(output, 1);

        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < input; j++) {
                weightsHidden.set(i, j, new Random().nextDouble());
            }
        }
        for (int i = 0; i < output; i++) {
            for (int j = 0; j < hidden; j++) {
                weightsOut.set(i, j, new Random().nextDouble());
            }
        }
        for (int i = 0; i < hidden; i++) {
            biasHidden.set(i, 0, new Random().nextDouble());
        }
        for (int i = 0; i < output; i++) {
            biasOut.set(i, 0, new Random().nextDouble());
        }
    }

    private void check(double[] X, double[] Y) {
        // forward
        SimpleMatrix input = new SimpleMatrix(new double[][]{X}).transpose(); // T(1xINP) = INPx1
        SimpleMatrix sumHidden = weightsHidden.mult(input).plus(biasHidden); // HIDxINP * INPx1 + HIDx1 = HIDx1
        hiddenOut = sigmoid(sumHidden); // HIDx1
        SimpleMatrix sumOut = weightsOut.mult(hiddenOut).plus(biasOut); // OUTxHID * HIDx1 + OUTx1
        SimpleMatrix out = sigmoid(sumOut);
        res = out.get(0, 0);
        double y = Y[0];
        double err = res - y;
        //String.format()
        //System.out.println("Res: " + out + " y: " + y + " Err: " + err);
        meanErr += Math.abs(err);
    }

    private void train(double[][] X, double[][] Y, int numberOfEpochs) {
        Random random = new Random();
        try {
            BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\p\\epoch.txt"));
            for (int i = 0; i < numberOfEpochs; i++) {
                int n = random.nextInt(X.length);
                run(X[n], Y[n]);
                if (i % (numberOfEpochs/20) == 0) {
                    System.out.println("Epoch: " + i);
                    System.out.println("Res: " + res + " Original: " + Y[n][0]);
                    test(testX, testY, false);
                    bf.write(i + " " + meanErr + "\n");
                }
            }
            bf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void run(double[] X, double[] Y) {
        // forward
        SimpleMatrix input = new SimpleMatrix(new double[][]{X}).transpose(); // T(1xINP) = INPx1
        SimpleMatrix sumHidden = weightsHidden.mult(input).plus(biasHidden); // HIDxINP * INPx1 + HIDx1 = HIDx1
        hiddenOut = sigmoid(sumHidden); // HIDx1
        SimpleMatrix sumOut = weightsOut.mult(hiddenOut).plus(biasOut); // OUTxHID * HIDx1 + OUTx1
        SimpleMatrix out = sigmoid(sumOut);
        res = out.get(0, 0);
        // back
        // out
        SimpleMatrix y = new SimpleMatrix(new double[][]{Y}).transpose(); // T(1xOUT) = OUTx1
        SimpleMatrix error = out.minus(y); // OUTx1 - OUTx1 = OUTx1
        SimpleMatrix gradOut = error.elementMult(dsigmoid(out)); // OUTx1 .* OUTx1 = OUTx1
        SimpleMatrix deltaWeightsOut = gradOut.mult(hiddenOut.transpose()).scale(learningRate); // OUTx1 * T(HIDx1) = OUTxHID
        weightsOut = weightsOut.minus(deltaWeightsOut); // OUTxHID - OUTxHID = OUTxHID
        biasOut = biasOut.minus(gradOut.scale(learningRate)); // OUTx1 - OUTx1 = OUTx1
        // hidden
        SimpleMatrix weightedGrad = weightsOut.transpose().mult(gradOut); // T(OUTxHID) * OUTx1 = HIDx1
        SimpleMatrix gradHidden = dsigmoid(hiddenOut).elementMult(weightedGrad); // HIDx1 .* HIDx1 = HIDx1
        SimpleMatrix deltaWeightsHidden = input.mult(gradHidden.transpose()).scale(learningRate); // INPx1 * T(HIDx1) = INPxHID
        weightsHidden = weightsHidden.minus(deltaWeightsHidden.transpose()); // HIDxINP - T(INPxHID) = HIDxINP
        biasHidden = biasHidden.minus(gradHidden.scale(learningRate)); // HIDx1 - HIDx1 = HIDx1
    }

    private SimpleMatrix sigmoid(SimpleMatrix m) {
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

    private SimpleMatrix dsigmoid(SimpleMatrix m) {
        int rows = m.numRows();
        int cols = m.numCols();
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = m.get(i, j);
                result[i][j] = v * (1. - v);
            }
        }
        return new SimpleMatrix(result);
    }

    private double sigmoid(double x) {
        return 1. / (1. + Math.exp(-x)); // 1 / (1 + e^-x)
    }

    private static double[][] readArrayFromFile(String fileName) {
        double[][] result = new double[0][0];
        try {
            FileReader filereader = new FileReader(fileName);
            BufferedReader bufferedreader = new BufferedReader(filereader);
            String line = bufferedreader.readLine();
            List<String> lines = new ArrayList<String>();
            //While we have read in a valid line
            while (line != null) {
                //Try to parse integer from the String line
                lines.add(line);
                line = bufferedreader.readLine();
            }
            result = new double[lines.size()][];
            for (int i = 0; i < lines.size(); i++) {
                double[] doubleValues = Arrays.stream(lines.get(i).split("\\s+"))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                result[i] = doubleValues;
            }
        } catch(Exception e ) {
            e.printStackTrace();
        }
        return result;
    }
}