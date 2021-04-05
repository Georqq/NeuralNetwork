import activationfunctions.*;
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

    static ActivationFunction af = new SigmoidActivationFunction();

    double res;
    double sumError = Double.MAX_VALUE;
    double meanError = Double.MAX_VALUE;

    double learningRate = 0.01;

    static double [][] X, Y, testX, testY;

    public static void main(String[] args) {
        for (double i = -7.; i < 100.; i += 0.1) {
            System.out.println(i + " " + af.f(i) + " " + af.df(i));
        }
        long t1 = System.nanoTime();
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 5, 1);
        double[][] data = readArrayFromFile("E:\\p\\Data\\train.dat");
        System.out.println(Arrays.deepToString(data));
        double[] coefficients = {50., 100., 45.}; //findCoefficientsForNormalization(data);
        double[][] normalizedData = normalize(data, coefficients);
        Pair pairOfArrays = split(normalizedData, 2);
        X = (double[][]) pairOfArrays.obj1;
        Y = (double[][]) pairOfArrays.obj2;

        data = readArrayFromFile("E:\\p\\Data\\test.dat");
        System.out.println(Arrays.deepToString(data));
        normalizedData = normalize(data, coefficients);
        pairOfArrays = split(normalizedData, 2);
        testX = (double[][]) pairOfArrays.obj1;
        testY = (double[][]) pairOfArrays.obj2;

        neuralNetwork.train(X, Y, 2_000_001, 0.001);
        long t2 = System.nanoTime();
        System.out.println("Time: " + (t2 - t1) / 1_000_000 + " ms");
        neuralNetwork.test(testX, testY, true);
    }

    private void test(double[][] testX, double[][] testY, boolean output) {
        sumError = 0.;
        for (int i = 0; i < testX.length; i++) {
            check(testX[i], testY[i]);
        }
        meanError = sumError / (double) testX.length;
        if (output) {
            System.out.println("MeanError: " + meanError);
        }
    }

    public NeuralNetwork(int input, int hidden, int output) {
        weightsHidden = new SimpleMatrix(hidden, input);
        weightsOut = new SimpleMatrix(output, hidden);
        biasHidden = new SimpleMatrix(hidden, 1);
        biasOut = new SimpleMatrix(output, 1);
        hiddenOut = new SimpleMatrix(hidden , 1);
        out = new SimpleMatrix(output, 1);

        Random random = new Random();

        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < input; j++) {
                weightsHidden.set(i, j, random.nextDouble());
            }
        }
        for (int i = 0; i < output; i++) {
            for (int j = 0; j < hidden; j++) {
                weightsOut.set(i, j, random.nextDouble());
            }
        }
        for (int i = 0; i < hidden; i++) {
            biasHidden.set(i, 0, random.nextDouble());
        }
        for (int i = 0; i < output; i++) {
            biasOut.set(i, 0, random.nextDouble());
        }
    }

    private void check(double[] X, double[] Y) {
        // forward
        SimpleMatrix input = new SimpleMatrix(new double[][]{X}).transpose(); // T(1xINP) = INPx1
        SimpleMatrix sumHidden = weightsHidden.mult(input).plus(biasHidden); // HIDxINP * INPx1 + HIDx1 = HIDx1
        hiddenOut = af.f(sumHidden); // HIDx1
        SimpleMatrix sumOut = weightsOut.mult(hiddenOut).plus(biasOut); // OUTxHID * HIDx1 + OUTx1
        SimpleMatrix out = af.f(sumOut);
        res = out.get(0, 0);
        double y = Y[0];
        double err = res - y;
        sumError += Math.abs(err);
    }

    private void train(double[][] X, double[][] Y, int numberOfEpochs, double meanEr) {
        Random random = new Random();
        try {
            BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\p\\Data\\epoch.txt"));
            for (int i = 0; i < numberOfEpochs && meanError > meanEr; i++) {
                int n = random.nextInt(X.length);
                run(X[n], Y[n]);
                if (i % (numberOfEpochs/25) == 0) {
                    System.out.println("Epoch: " + i + "\tMean Error: " + meanError);
                    test(testX, testY, false);
                    bf.write(i + " " + meanError + "\n");
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
        hiddenOut = af.f(sumHidden); // HIDx1
        SimpleMatrix sumOut = weightsOut.mult(hiddenOut).plus(biasOut); // OUTxHID * HIDx1 + OUTx1
        SimpleMatrix out = af.f(sumOut);
        res = out.get(0, 0);
        // back
        // out
        SimpleMatrix y = new SimpleMatrix(new double[][]{Y}).transpose(); // T(1xOUT) = OUTx1
        SimpleMatrix error = out.minus(y); // OUTx1 - OUTx1 = OUTx1
        SimpleMatrix gradOut = error.elementMult(af.dfOut(out)); // OUTx1 .* OUTx1 = OUTx1
        SimpleMatrix deltaWeightsOut = gradOut.mult(hiddenOut.transpose()).scale(learningRate); // OUTx1 * T(HIDx1) = OUTxHID
        weightsOut = weightsOut.minus(deltaWeightsOut); // OUTxHID - OUTxHID = OUTxHID
        biasOut = biasOut.minus(gradOut.scale(learningRate)); // OUTx1 - OUTx1 = OUTx1
        // hidden
        SimpleMatrix weightedGrad = weightsOut.transpose().mult(gradOut); // T(OUTxHID) * OUTx1 = HIDx1
        SimpleMatrix gradHidden = af.dfOut(hiddenOut).elementMult(weightedGrad); // HIDx1 .* HIDx1 = HIDx1
        SimpleMatrix deltaWeightsHidden = input.mult(gradHidden.transpose()).scale(learningRate); // INPx1 * T(HIDx1) = INPxHID
        weightsHidden = weightsHidden.minus(deltaWeightsHidden.transpose()); // HIDxINP - T(INPxHID) = HIDxINP
        biasHidden = biasHidden.minus(gradHidden.scale(learningRate)); // HIDx1 - HIDx1 = HIDx1
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

    public static double[] findCoefficientsForNormalization(double[][] array) {
        double[] coefficients = new double[array[0].length];
        for (double[] doubles : array) {
            for (int j = 0; j < array[0].length; j++) {
                if (doubles[j] > coefficients[j]) {
                    coefficients[j] = doubles[j];
                }
            }
        }
        return coefficients;
    }

    public static double[][] normalize(double[][] array, double[] coeffs) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                array[i][j] /= coeffs[j];
            }
        }
        return array;
    }

    public static Pair split(double[][] data, int leftArrayLength) {
        double[][] left = new double[data.length][leftArrayLength];
        double[][] right = new double[data.length][data[0].length - leftArrayLength];

        for (int i = 0; i < data.length; i++) {
            if (leftArrayLength >= 0) System.arraycopy(data[i], 0, left[i], 0, leftArrayLength);
            if (data[0].length - leftArrayLength >= 0)
                System.arraycopy(data[i], leftArrayLength, right[i], 0, data[0].length - leftArrayLength);
        }
        return new Pair(left, right);
    }
}