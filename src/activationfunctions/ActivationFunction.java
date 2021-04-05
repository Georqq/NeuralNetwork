package activationfunctions;

import org.ejml.simple.SimpleMatrix;

public interface ActivationFunction {
    double f(double x);
    double df(double x);
    double dfOut(double output);
    SimpleMatrix f(SimpleMatrix m);
    SimpleMatrix df(SimpleMatrix m);
    SimpleMatrix dfOut(SimpleMatrix output);
}
