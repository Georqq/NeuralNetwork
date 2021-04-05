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

    double sumError = Double.MAX_VALUE;
    double meanError = Double.MAX_VALUE;

    double learningRate = 0.01;

    static double [][] X, Y, testX, testY;

    public static void main(String[] args) {
        long t1 = System.nanoTime();
        //testRun();
        fillMNISTdata();
        NeuralNetwork neuralNetwork = new NeuralNetwork(784, 80, 10);
        neuralNetwork.train(X, Y, 10_000_001, 1E-20);
        long t2 = System.nanoTime();
        System.out.println("Time: " + (t2 - t1) / 1_000_000 + " ms");
        neuralNetwork.test(testX, testY, true);
    }

    private static void fillTestData() {
        double[][] data = readArrayFromFile("E:\\p\\Data\\train.dat", "\\s+");
        double[] coefficients = {50., 100., 45.}; //findCoefficientsForNormalization(data);
        double[][] normalizedData = normalize(data, coefficients);
        Pair pairOfArrays = split(normalizedData, 2);
        Y = (double[][]) pairOfArrays.obj2;
        X = (double[][]) pairOfArrays.obj1;
        System.out.println(Arrays.toString(Y[0]));
        System.out.println(Arrays.toString(Y[1]));
        System.out.println(Arrays.toString(Y[2]));

        data = readArrayFromFile("E:\\p\\Data\\test.dat", "\\s+");
        //data = readArrayFromFile("E:\\p\\Data\\mnist_test.csv", ",");
        normalizedData = normalize(data, coefficients);
        pairOfArrays = split(normalizedData, 2);
        testX = (double[][]) pairOfArrays.obj1;
        testY = (double[][]) pairOfArrays.obj2;
    }

    private static void fillMNISTdata() {
        double[][] data = readArrayFromFile("E:\\p\\Data\\mnist_train.csv", ",");
        System.out.println(data.length + " " + data[0].length);
        double[] coefficients = new double[785];
        coefficients[0] = 1.;
        for (int i = 1; i < coefficients.length; i++) {
            coefficients[i] = 255.;
        }
        double[][] normalizedData = normalize(data, coefficients);
        Pair pairOfArrays = split(normalizedData, 1);
        Y = (double[][]) pairOfArrays.obj1;
        X = (double[][]) pairOfArrays.obj2;
        double[][] tempY = new double[60000][10];
        for (int i = 0; i < tempY.length; i++) {
            int index = (int) Y[i][0];
            tempY[i][index] = 1.;
        }
        Y = tempY;
        System.out.println(Arrays.toString(Y[0]));
        System.out.println(Arrays.toString(Y[1]));
        System.out.println(Arrays.toString(Y[2]));

        data = readArrayFromFile("E:\\p\\Data\\mnist_test.csv", ",");
        normalizedData = normalize(data, coefficients);
        pairOfArrays = split(normalizedData, 1);
        testX = (double[][]) pairOfArrays.obj2;
        testY = (double[][]) pairOfArrays.obj1;
        tempY = new double[10000][10];
        for (int i = 0; i < tempY.length; i++) {
            int index = (int) Y[i][0];
            tempY[i][index] = 1.;
        }
        testY = tempY;
    }

    private void test(double[][] testX, double[][] testY, boolean output) {
        sumError = 0.;
        try {
            BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\p\\Data\\check.txt"));
            for (int i = 0; i < testY.length; i++) {
                bf.write(Arrays.toString(testY[i]) + " ");
                bf.write(Arrays.toString(check(testX[i], testY[i])) + "\n");
            }
            meanError = sumError / (double) testX.length / 2.;
            if (output) {
                System.out.println("MeanError: " + meanError);
            }
            bf.close();
        } catch (IOException ignored) { }
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

    private double[] check(double[] X, double[] Y) {
        // forward
        SimpleMatrix input = new SimpleMatrix(new double[][]{X}).transpose(); // T(1xINP) = INPx1
        SimpleMatrix sumHidden = weightsHidden.mult(input).plus(biasHidden); // HIDxINP * INPx1 + HIDx1 = HIDx1
        hiddenOut = af.f(sumHidden); // HIDx1
        SimpleMatrix sumOut = weightsOut.mult(hiddenOut).plus(biasOut); // OUTxHID * HIDx1 + OUTx1
        SimpleMatrix out = af.f(sumOut);
        double[] result = new double[out.numRows()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i, 0);
        }
        /*
        res = out.get(0, 0);
        double y = Y[0];
         */
        SimpleMatrix y = new SimpleMatrix(new double[][]{Y}).transpose(); // T(1xOUT) = OUTx1
        SimpleMatrix error = out.minus(y); // OUTx1 - OUTx1 = OUTx1
        sumError = error.elementPower(2).elementSum() / (2. * error.numRows());
        //sumError += Math.abs(err);
        return result;
    }

    private void train(double[][] X, double[][] Y, int numberOfEpochs, double meanEr) {
        Random random = new Random();
        try {
            BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\p\\Data\\epoch.txt"));
            for (int i = 0; i < numberOfEpochs && meanError > meanEr; i++) {
                int n = random.nextInt(X.length);
                run(X[n], Y[n]);
                if (i % (numberOfEpochs/1000) == 0) {
                    System.out.println("Epoch: " + i + "\tMean Error: " + meanError);
                    //test(testX, testY, false);
                    //System.out.println(Arrays.toString(X[n]));
                    //System.out.println(Arrays.toString(Y[n]));
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
        //res = out.get(0, 0);
        // back
        // out
        SimpleMatrix y = new SimpleMatrix(new double[][]{Y}).transpose(); // T(1xOUT) = OUTx1
        SimpleMatrix error = out.minus(y); // OUTx1 - OUTx1 = OUTx1
        meanError = error.elementPower(2).elementSum() / (2. * error.numRows());
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

    private static double[][] readArrayFromFile(String fileName, String regExSep) {
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
                double[] doubleValues = Arrays.stream(lines.get(i).split(regExSep))
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